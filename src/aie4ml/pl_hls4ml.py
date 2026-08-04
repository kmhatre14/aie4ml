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

# op_type (aie4ml) / hls4ml class_name -> the nnet:: streaming function the wrapper calls.
# Each entry is a weight-less op called as fn<input_t, result_t, CONFIG>(in_stream, out_stream).
_NNET_CALL = {
    'Softmax': 'nnet::softmax',
    'softmax': 'nnet::softmax',
}

_DEFAULT_PART = 'xcve2802-vsvh1760-2MP-e-S'  # VEK280; only labels the hls4ml project (we do not build it)
_FIFO_MAX_BITS = 4096  # hls::stream element aggregate limit


def _parse_fixed(prec: str):
    """'ap_ufixed<36,16,...>' | 'ufixed<36,16>' -> (unsigned: bool, W: int, I: int)."""
    m = re.search(r'(ap_)?(u?)fixed<\s*(\d+)\s*,\s*(-?\d+)', str(prec))
    if not m:
        raise ValueError(f'cannot parse fixed-point precision {prec!r}')
    return m.group(2) == 'u', int(m.group(3)), int(m.group(4))


def _fmt_qi(qi) -> str:
    """aie4ml QuantIntent (width, frac, signed) -> hls4ml precision string 'fixed<W,I>'/'ufixed<W,I>'
    (I = integer bits = width - frac)."""
    w, frac = int(qi.width), int(qi.frac)
    return f"{'' if qi.signed else 'u'}fixed<{w},{w - frac}>"


def _fit_fifo(width: int, int_bits: int, n_feat: int):
    """Narrow a fixed-point width so n_feat elements fit the <=4096-bit stream limit, keeping the
    integer bits (value range) and trimming fractional bits. Returns (W, I)."""
    if n_feat * width <= _FIFO_MAX_BITS:
        return width, int_bits
    cap = _FIFO_MAX_BITS // n_feat
    new_w = cap
    new_i = min(int_bits, new_w)
    return new_w, new_i


def generate_pl_kernel(model_or_ctx, *, name, source_layer, beats_per_iter, out_dir) -> Dict[str, Any]:
    """Slice ``source_layer`` out of the retained hls4ml ModelGraph, write its HLS firmware under
    ``<out_dir>/pl/<name>_hls/``, and return the wrapper template vars."""
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

    class_name = node.class_name
    nnet_fn = _NNET_CALL.get(class_name)
    if nnet_fn is None:
        raise NotImplementedError(
            f'{layer}: auto PL kernel for op {class_name!r} is not supported yet (v1 supports '
            f'weight-less streaming ops: {sorted(set(_NNET_CALL))}).'
        )

    in_var = node.get_input_variable()
    in_name = node.inputs[0]
    n_feat = int(in_var.shape[-1])

    # ON-WIRE precisions come from the aie4ml IR cut (the PLIO dtype authority), NOT the ModelGraph
    # node: the AIE-flow node carries internal accumulator precisions (e.g. fixed<41,20>), while the
    # cut carries the quantized int8/uint8 the data movers actually transfer across the PLIO.
    log = ctx.ir.logical
    cut = next((c for c in log.pl_cuts if c.source_layer == layer), None)
    if cut is None:
        raise RuntimeError(f'no PL cut recorded for layer {layer!r}')
    in_prec = _fmt_qi(log.tensors[cut.cut_out_tensor].precision)   # AIE -> PL (op input)
    u, w, i = _parse_fixed(_fmt_qi(log.tensors[cut.cut_in_tensor].precision))  # PL -> AIE (op output)
    w2, i2 = _fit_fifo(w, i, n_feat)
    out_frac = w2 - i2
    out_prec = f"{'u' if u else ''}fixed<{w2},{i2}>"

    proj = f'{name}_hls'
    proj_dir = Path(out_dir) / 'pl' / proj

    sub_cfg = {
        'OutputDir': str(proj_dir),
        'ProjectName': proj,
        'Backend': 'Vitis',
        # A concrete Vitis part (not the aie4ml platform string, which hls4ml's config parser would
        # reject). Only labels the sub-project; aie4ml builds the .xo itself, never hls4ml's build.
        'Part': _DEFAULT_PART,
        'IOType': 'io_stream',
        'HLSConfig': {
            'Model': {'Precision': 'fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
            'LayerName': {
                in_name: {'Precision': {'result': in_prec}},
                layer: {
                    'Precision': {'result': out_prec},
                    'implementation': 'stable',
                    'exp_table_t': 'fixed<18,8,RND,SAT>',
                    'inv_table_t': 'fixed<18,8,RND,SAT>',
                },
            },
        },
    }
    input_layer = {'class_name': 'InputLayer', 'name': in_name, 'input_shape': [n_feat], 'outputs': [in_name]}
    op_layer = {
        'class_name': class_name, 'name': layer, 'inputs': [in_name], 'outputs': [layer],
        'activation': 'softmax', 'axis': -1, 'n_in': n_feat,
    }
    sub = ModelGraph.from_layer_list(sub_cfg, [input_layer, op_layer], inputs=[in_name], outputs=[layer])
    sub.write()

    params_h = proj_dir / 'firmware' / 'parameters.h'
    cfg_name = _config_struct_name(params_h)

    # rows one PLIO stream carries: beats_per_iter beats / (n_feat/lanes) beats-per-row.
    lanes_per_beat = int(ctx.device.plio_width_bits) // int(spec_width(ctx, layer))
    beats_per_row = n_feat // lanes_per_beat
    rows_per_stream = beats_per_iter // beats_per_row
    lane_bits = int(spec_width(ctx, layer))

    return {
        'hls_proj': proj,                       # firmware dir + include prefix
        'hls_defines': f'{proj}/firmware/defines.h',
        'hls_params': f'{proj}/firmware/parameters.h',
        'nnet_fn': nnet_fn,                      # e.g. nnet::softmax
        'hls_cfg': cfg_name,                     # e.g. softmax_config2 (from the generated file)
        'feats': n_feat,
        'rows_per_stream': rows_per_stream,
        'beats_per_row': beats_per_row,
        'lanes_per_beat': lanes_per_beat,
        'lane_bits': lane_bits,
        'trunc_hi': out_frac - 1,               # uint8 = top lane_bits of the fractional field
        'trunc_lo': out_frac - lane_bits,
    }


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
