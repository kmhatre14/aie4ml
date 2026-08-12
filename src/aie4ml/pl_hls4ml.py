# Copyright 2026 Advent Lab, aie4ml

"""Generate a PL kernel for an offloaded layer by slicing the retained hls4ml ModelGraph and
re-invoking hls4ml on the sub-graph. hls4ml is used UNMODIFIED, as a library:

    sub = ModelGraph.from_layer_list(sub_cfg, [InputLayer, <op>], ...)   # hls4ml public API
    sub.write()   ->  <out>/pl/<name>_hls/firmware/{defines.h, parameters.h, nnet_utils/, weights/}

The AXIS wrapper (templates/firmware/pl/compute/hls4ml_kernel.cpp.jinja) then #includes that
firmware and calls the op's nnet:: function with a derived config that batches n_in over the rows
one PLIO stream carries -- reproducing, automatically, the hand-graft in proj_softmax_pl_merge.

v1 supports a single streaming (weight-less) op per cut (softmax/activation). Weighted ops and
multi-layer sub-graphs are later phases (see PLAN_hls4ml_pl_fork.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from .ir import get_backend_context

# aie4ml op key -> how to slice it and call it in the wrapper.
_OP_SPEC = {
    'softmax':    {'nnet_fn': 'nnet::softmax', 'size_field': 'n_in',   'n_inputs': 1},
    # LayerNorm is DIFFERENT: io_parallel (array in/out, not streams), weighted (gamma/beta), and
    # the call-the-top wrapper (hls4ml_baked_weights.cpp.jinja). Handled by a dedicated branch below.
    'layer_norm': {'nnet_fn': 'nnet::layernormalize', 'size_field': 'n_in', 'n_inputs': 1,
                   'io': 'parallel', 'weighted': True},
}

_FIFO_MAX_BITS = 4096  # hls::stream element aggregate limit


def _pl_part(ctx) -> str:
    """
    Vitis part for the hls4ml PL offload
    """
    part = getattr(ctx.device, 'part', '') or ''
    if not part:
        raise RuntimeError(
            f'PL offload needs a Vitis "Part" for platform {ctx.device.platform!r}, but its '
            f'aie_devices.json entry has none. Add a "Part" field (e.g. "xcve2802-vsvh1760-2MP-e-S") '
            f'to that device entry.'
        )
    return part


def _pl_reuse_factor(mg, node) -> int:
    """
    ReuseFactor for this layer's hls4ml offloads. Defaults to 1 (fully spatial). 
    """
    raw = mg.config.get_layer_config_value(node, 'pl_reuse_factor', 1)
    try:
        rf = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{node.name}: pl_reuse_factor must be a positive integer, got {raw!r}.')
    if rf < 1:
        raise ValueError(f'{node.name}: pl_reuse_factor must be >= 1, got {rf}.')
    return rf


def _format_quant_intent(qi) -> str:
    """aie4ml QuantIntent (width, frac, signed) -> hls4ml precision string 'fixed<W,I>'/'ufixed<W,I>'
    (I = integer bits = width - frac)."""
    w, frac = int(qi.width), int(qi.frac)
    return f"{'' if qi.signed else 'u'}fixed<{w},{w - frac}>"


def _op_key(node) -> str:
    """Normalize a retained-ModelGraph node to an aie4ml op key ('softmax' | 'layer_norm'), else None."""
    cn = node.class_name
    if cn in ('Softmax', 'softmax'):
        return 'softmax'
    if cn == 'LayerNormalization':
        return 'layer_norm'
    return None


def generate_pl_kernel(
        model_or_ctx, *, 
        name, 
        source_layer, 
        beats_per_iter, 
        n_op_inputs, 
        shards_per_input, 
        out_dir
    ) -> Dict[str, Any]:
    
    """Slice ``source_layer`` out of the retained hls4ml ModelGraph, write its HLS firmware under
    ``<out_dir>/pl/<name>_hls/``, and return the wrapper template vars. Handles the streaming
    reduction op (softmax, batch-split) and the io_parallel weighted op (LayerNorm)."""
    import hls4ml  # noqa: F401  (import guarded here so aie-only flows need not import hls4ml)
    from hls4ml.model.graph import ModelGraph

    ctx = get_backend_context(model_or_ctx)
    mg = ctx.source_model
    if mg is None:
        raise RuntimeError(
            'PL offload via hls4ml needs the source hls4ml ModelGraph, but ctx.source_model is None. '
            '(Only the hls4ml frontend retains it; ONNX/other frontends cannot use auto PL kernels yet.)'
        )

    layer = source_layer
    node = getattr(mg, 'graph', {}).get(layer)
    if node is None:
        raise RuntimeError(f'PL offload: layer {layer!r} not found in the retained hls4ml ModelGraph.')

    opkey = _op_key(node)
    op = _OP_SPEC.get(opkey)
    if op is None:
        raise NotImplementedError(
            f'{layer}: auto PL kernel for op {node.class_name!r} is not supported yet '
            f'(supported: {sorted(_OP_SPEC)}).'
        )
    if op['n_inputs'] != int(n_op_inputs):
        raise RuntimeError(f'{layer}: op {opkey!r} takes {op["n_inputs"]} input(s) but the cut has {n_op_inputs}.')

    if op.get('io') == 'parallel':  # LayerNorm: io_parallel + baked gamma/beta (call-the-top)
        return _layernorm_kernel(ctx, mg, node, name=name, source_layer=layer,
                                 beats_per_iter=beats_per_iter, shards_per_input=shards_per_input, out_dir=out_dir)

    log = ctx.ir.logical
    cut = next((c for c in log.pl_cuts if c.source_layer == layer), None)
    if cut is None:
        raise RuntimeError(f'no PL cut recorded for layer {layer!r}')

    full_feat = int(node.get_input_variable().shape[-1])
    # PER-STREAM feature count: a batch-split (reduction) op carries WHOLE rows on each stream; a
    # feature-split (elementwise) op carries a 1/shards slice. The sub-model + wrapper are built at
    # this per-stream width.
    per_stream_feat = full_feat if cut.reduces_features else full_feat // int(shards_per_input)
    if per_stream_feat < 1:
        raise RuntimeError(f'{layer}: per-stream feature count < 1 (full={full_feat}, shards={shards_per_input}).')

    in_precs = [_format_quant_intent(log.tensors[t].precision) for t in cut.cut_out_tensors]  # per operand, in order
    out_prec = _format_quant_intent(log.tensors[cut.cut_in_tensor].precision)
    lane_bits = int(cut.width)
    if per_stream_feat * lane_bits > _FIFO_MAX_BITS:
        raise NotImplementedError(
            f'{layer}: per-stream array is {per_stream_feat}x{lane_bits}={per_stream_feat * lane_bits} bits '
            f'> {_FIFO_MAX_BITS}-bit hls::stream limit; needs more shards or a narrower dtype.'
        )

    proj = f'{name}_hls'
    proj_dir = Path(out_dir) / 'pl' / proj

    # Sub-model: N InputLayers -> op. Built at the per-stream feature width.
    input_names = [f'{layer}_in{k}' for k in range(op['n_inputs'])]
    layer_cfg = {nm: {'Precision': {'result': in_precs[k]}} for k, nm in enumerate(input_names)}
    # Only the streaming softmax reaches here (layer_norm returns above; unsupported ops already raised).
    layer_cfg[layer] = {
        'Precision': {'result': out_prec}, 'implementation': 'stable',
        'exp_table_t': 'fixed<18,8,RND,SAT>', 'inv_table_t': 'fixed<18,8,RND,SAT>',
    }
    op_layer = {'class_name': 'Softmax', 'name': layer, 'inputs': input_names, 'outputs': [layer],
                'activation': 'softmax', 'axis': -1, 'n_in': per_stream_feat}
    layer_list = [
        {'class_name': 'InputLayer', 'name': nm, 'input_shape': [per_stream_feat], 'outputs': [nm]}
        for nm in input_names
    ]
    layer_list.append(op_layer)

    sub_cfg = {
        'OutputDir': str(proj_dir),
        'ProjectName': proj,
        'Backend': 'Vitis',
        # A concrete Vitis part (not the aie4ml platform string, which hls4ml's config parser would
        # reject). Sourced from the device catalog ("Part" in aie_devices.json); only labels the
        # sub-project -- aie4ml builds the .xo itself, never hls4ml's build.
        'Part': _pl_part(ctx),
        'IOType': 'io_stream',
        'HLSConfig': {'Model': {'Precision': 'fixed<16,6>', 'ReuseFactor': _pl_reuse_factor(mg, node),
                                'Strategy': 'Latency'},
                      'LayerName': layer_cfg},
    }
    sub = ModelGraph.from_layer_list(sub_cfg, layer_list, inputs=input_names, outputs=[layer])
    sub.write()

    cfg_name = _config_struct_name(proj_dir / 'firmware' / 'parameters.h')
    lanes_per_beat = int(ctx.device.plio_width_bits) // lane_bits
    beats_per_row = per_stream_feat // lanes_per_beat
    rows_per_stream = beats_per_iter // beats_per_row

    return {
        'hls_proj': proj,                       # firmware dir + include prefix
        'hls_defines': f'{proj}/firmware/defines.h',
        'hls_params': f'{proj}/firmware/parameters.h',
        'nnet_fn': op['nnet_fn'],               # nnet::softmax
        'hls_cfg': cfg_name,                    # generated config struct (e.g. softmax_config2)
        'size_field': op['size_field'],         # config field to batch over rows (n_in)
        'n_op_inputs': op['n_inputs'],          # operand streams per output shard (1 for softmax)
        'feats': per_stream_feat,
        'rows_per_stream': rows_per_stream,
        'beats_per_row': beats_per_row,
        'lanes_per_beat': lanes_per_beat,
        'lane_bits': lane_bits,
        'trunc_hi': lane_bits - 1,              # result element is exactly lane_bits wide -> copy all
        'trunc_lo': 0,
    }


def _layernorm_kernel(ctx, mg, node, *, name, source_layer, beats_per_iter, shards_per_input, out_dir):
    """LayerNorm PL kernel: io_parallel nnet::layernormalize with gamma/beta BAKED into the .xo via
    the hls4ml-generated top (call-the-top). hls4ml requires a 3-D LN input, so the sub-model uses
    shape [1, feat]; the trained gamma/beta are baked into the sub-model's firmware."""
    import numpy as np
    from hls4ml.model.graph import ModelGraph

    layer = source_layer
    log = ctx.ir.logical
    cut = next((c for c in log.pl_cuts if c.source_layer == layer), None)
    if cut is None:
        raise RuntimeError(f'no PL cut recorded for layer {layer!r}')

    full_feat = int(node.get_input_variable().shape[-1])
    per_stream_feat = full_feat if cut.reduces_features else full_feat // int(shards_per_input)
    lane_bits = int(cut.width)
    if per_stream_feat * lane_bits > _FIFO_MAX_BITS:
        raise NotImplementedError(
            f'{layer}: per-stream array {per_stream_feat}x{lane_bits} > {_FIFO_MAX_BITS}-bit FIFO limit.'
        )

    in_prec = _format_quant_intent(log.tensors[cut.cut_out_tensors[0]].precision)
    out_prec = _format_quant_intent(log.tensors[cut.cut_in_tensor].precision)

    w = getattr(node, 'weights', {}) or {}
    gamma = (np.asarray(w['scale'].data, dtype=np.float32).flatten() if 'scale' in w
             else np.ones(per_stream_feat, dtype=np.float32))
    beta = (np.asarray(w['bias'].data, dtype=np.float32).flatten() if 'bias' in w
            else np.zeros(per_stream_feat, dtype=np.float32))

    proj = f'{name}_hls'
    proj_dir = Path(out_dir) / 'pl' / proj
    in_name = f'{layer}_in0'
    sub_cfg = {
        'OutputDir': str(proj_dir), 'ProjectName': proj, 'Backend': 'Vitis', 'Part': _pl_part(ctx),
        'IOType': 'io_parallel',
        'HLSConfig': {'Model': {'Precision': 'fixed<16,6>', 'ReuseFactor': _pl_reuse_factor(mg, node),
                                'Strategy': 'Latency'},
                      'LayerName': {in_name: {'Precision': {'result': in_prec}},
                                    layer: {'Precision': {'result': out_prec}}}},
    }
    input_layer = {'class_name': 'InputLayer', 'name': in_name, 'input_shape': [1, per_stream_feat],
                   'outputs': [in_name]}
    ln_layer = {'class_name': 'LayerNormalization', 'name': layer, 'inputs': [in_name], 'outputs': [layer],
                'n_in': per_stream_feat, 'seq_len': 1, 'epsilon': 1e-3,
                'gamma_data': gamma, 'beta_data': beta, 'scale_data': gamma, 'bias_data': beta}
    sub = ModelGraph.from_layer_list(sub_cfg, [input_layer, ln_layer], inputs=[in_name], outputs=[layer])
    sub.write()

    cfg_name = _config_struct_name(proj_dir / 'firmware' / 'parameters.h')
    lanes_per_beat = int(ctx.device.plio_width_bits) // lane_bits
    beats_per_row = per_stream_feat // lanes_per_beat
    rows_per_stream = beats_per_iter // beats_per_row
    common = {
        'hls_proj': proj, 'hls_defines': f'{proj}/firmware/defines.h',
        'hls_params': f'{proj}/firmware/parameters.h', 'hls_cfg': cfg_name,
        'feats': per_stream_feat, 'rows_per_stream': rows_per_stream, 'beats_per_row': beats_per_row,
        'lanes_per_beat': lanes_per_beat, 'lane_bits': lane_bits,
    }
    # Weights (gamma/beta) are BAKED into the .xo: the hls4ml-generated top declares them as const
    # arrays and passes them to nnet::layernormalize internally, so the wrapper is pure data glue --
    # the same call-the-top flow works for any weighted op.
    # syn_freqhz over-constrains HLS (350 MHz) so the deep reduce/normalize datapath registers deeper
    # and closes timing at the real 312.5 MHz PL clock (the DATAFLOW wrapper leaves latency headroom).
    # Consumed by Makefile.jinja.
    return {**common, 'cpp_template': 'pl/compute/hls4ml_baked_weights.cpp.jinja',
            'hls_top': proj, 'hls_top_src': f'{proj}/firmware/{proj}.cpp',
            'syn_freqhz': 350000000}


def _config_struct_name(params_h: Path) -> str:
    txt = params_h.read_text()
    m = re.search(r'struct\s+(\w*config\d+)\s*:', txt)
    if not m:
        raise RuntimeError(f'could not find the op config struct in {params_h}')
    return m.group(1)


def spec_width(model_or_ctx, layer: str) -> int:
    """The cut boundary element bit-width (AXIS lane width) for this layer's cut."""
    ctx = get_backend_context(model_or_ctx)
    for cut in ctx.ir.logical.pl_cuts:
        if cut.source_layer == layer:
            return int(cut.width)
    raise RuntimeError(f'no PL cut recorded for layer {layer!r}')
