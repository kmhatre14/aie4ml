# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Lower hls4ml model graphs to the AIE IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ...device_catalog import load_device_catalog
from ...ir import (
    BackendPolicies,
    LogicalIR,
    OpNode,
    TensorVar,
    ensure_backend_context,
    set_input_roles,
)
from ...ir.context import AIEBackendContext, DeviceSpec, ProjectConfig
from ...passes.utils import is_pointwise_dense
from ..common import register_default_traits
from .utils import _create_weight_tensors, _get_post_activation_precision, _precision_of, extract_layer_directives


class LowerToAieIr(ModelOptimizerPass):
    """Build the shared IR graph from the frontend model."""

    def __init__(self):
        self.name = 'lower_to_aie_ir'

    def transform(self, model) -> bool:
        ctx = ensure_backend_context(model, lambda: self._create_context(model))
        ctx.reset_ir()
        # Retain the hls4ml ModelGraph so the PL-offload path can slice marked layers back out and
        # re-invoke hls4ml on the sub-graph to generate the PL kernel (pl_hls4ml.py).
        ctx.source_model = model

        graph: LogicalIR = ctx.ir.logical
        layers = list(model.get_layers())
        input_var = model.get_input_variables()[0]
        batch_size = int(model.config.get_config_value('AIEConfig', {})['BatchSize'])
        batch_included = bool(ctx.policies.tensors_have_batch)

        def _canon(shape):
            dims = [int(x) for x in shape]
            return tuple(dims) if batch_included else tuple([batch_size] + dims)

        if input_var.name not in graph.tensors:
            graph.add_tensor(
                TensorVar(
                    name=input_var.name,
                    shape=input_var.shape,
                    precision=_precision_of(input_var),
                )
            )
        graph.mark_graph_input(input_var.name)

        # LayerNorm reshape unwrap: hls4ml's LayerNormalization requires a 3-D (batch, seq, feat)
        # input, so models wrap it in reshapes. We drop ONLY the reshapes directly adjacent to a 
        # LayerNormalization -- the one feeding it and the one consuming its output
        # Note: This case is handled because hls4ml does not have 2D layernorm support during parsing.
        # This will not apply to onnx based frontends.
        reshape_tensor_redirect: Dict[str, str] = {}   # folded-away tensor name -> surviving tensor name
        bypassed_reshapes: set = set()       # reshape layer names that never become AIE nodes

        # Collect, in one pass: the LayerNorm layer names, and the reshapes each LayerNorm consumes as input.
        layernorm_names: set = set()
        reshapes_feeding_layernorm: set = set()
        for layer in layers:
            if layer.class_name == 'LayerNormalization':
                layernorm_names.add(layer.name)
                reshapes_feeding_layernorm.update(layer.inputs)

        for layer in layers:
            if layer.class_name != 'Reshape':
                continue
            feeds_layernorm = layer.name in reshapes_feeding_layernorm    # reshape BEFORE a LayerNorm
            follows_layernorm = layer.inputs[0] in layernorm_names        # reshape AFTER a LayerNorm
            if not feeds_layernorm and not follows_layernorm:
                continue

            in_src = layer.inputs[0]
            in_var = input_var if in_src == 'input' else model.output_vars[in_src]
            out_var = model.output_vars[layer.name]
            bypassed_reshapes.add(layer.name)
            # Keep the LOWER-rank (2-D) tensor; drop the higher-rank one so the graph stays 2-D.
            if len(out_var.shape) <= len(in_var.shape):
                reshape_tensor_redirect[in_var.name] = out_var.name   # flatten (1,feat) -> (feat)
            else:
                reshape_tensor_redirect[out_var.name] = in_var.name   # expand  (feat) -> (1,feat)

        def _resolve(name: str) -> str:
            seen: set = set()
            while name in reshape_tensor_redirect and name not in seen:
                seen.add(name)
                name = reshape_tensor_redirect[name]
            return name

        for layer in layers:
            var = model.output_vars[layer.name]
            if var.name in reshape_tensor_redirect:
                continue  # bypassed no-op reshape -- no tensor; consumers resolve to its input
            if var.name not in graph.tensors:
                prec = _precision_of(var)
                if layer.class_name == 'Dense' or is_pointwise_dense(layer):
                    # hls4ml may remove the following linear activation before lowering,
                    # leaving the Dense output var at accumulator precision.  Recover the
                    # correct post-activation precision from the config if possible.
                    override = _get_post_activation_precision(layer, model)
                    if override is not None:
                        prec = override
                graph.add_tensor(
                    TensorVar(
                        name=var.name,
                        shape=_canon(var.shape),
                        precision=prec,
                    )
                )

        node_map: Dict[str, OpNode] = {}
        param_tensors: Dict[str, tuple] = {}
        created_nodes = set()

        for layer in layers:
            if layer.name in bypassed_reshapes:
                continue  # no-op reshape -- never becomes an AIE node (its tensor is aliased)
            node = OpNode(
                name=f'{layer.name}_aie',
                op_type=self._map_op_type(layer),
                dialect=ctx.device.dialect,
            )
            if layer.class_name.lower() == 'input':
                node.is_placeholder = True
            self._collect_metadata(layer, node)
            node.directives.update(extract_layer_directives(layer, model))

            if node.op_type == 'dense':
                weight_tv, bias_tv = _create_weight_tensors(layer, graph)
                param_tensors[layer.name] = (weight_tv, bias_tv)

            if node.op_type == 'layer_norm':
                for weight_name, role in (('scale', 'gamma'), ('bias', 'beta')):
                    wvar = layer.weights.get(weight_name)
                    if wvar is None:
                        raise RuntimeError(f'{layer.name}: LayerNormalization missing {weight_name!r} weight.')
                    intent = _precision_of(wvar)
                    tv = TensorVar(
                        name=f'{layer.name}_{role}',
                        shape=np.asarray(wvar.data).shape,
                        precision=intent,
                        data=wvar.data,
                    )
                    graph.add_tensor(tv)
                    if weight_name == 'scale':
                        gamma_tv = tv
                    else:
                        beta_tv = tv
                param_tensors[layer.name] = (gamma_tv, beta_tv)

            var = model.output_vars[layer.name]
            tv = graph.tensors[_resolve(var.name)]  # flatten reshape: emit the low-rank output
            tv.producer = node
            node.outputs.append(tv)
            if node.op_type == 'transpose':
                self._normalize_transpose_perm(node, tv.shape)
            graph.add_node(node)
            node_map[layer.name] = node
            created_nodes.add(layer.name)

        for layer in layers:
            if layer.name not in created_nodes:
                continue
            node = node_map[layer.name]
            if layer.class_name.lower() == 'input':
                continue

            for src in layer.inputs:
                var = input_var if src == 'input' else model.output_vars[src]
                tv = graph.tensors[_resolve(var.name)]  # resolve through bypassed no-op reshapes
                node.inputs.append(tv)
                tv.consumers.append(node)

            if layer.name in param_tensors:
                weight_tv, bias_tv = param_tensors[layer.name]
                node.inputs.append(weight_tv)
                if bias_tv is not None:
                    node.inputs.append(bias_tv)

            role_names = list(node.metadata.get('input_roles') or [])
            if role_names:
                set_input_roles(node, node.inputs, role_names)

        for out_var in model.get_output_variables():
            graph.mark_graph_output(_resolve(out_var.name))

        return True

    def _collect_metadata(self, layer, node) -> None:
        meta: Dict[str, Any] = {}

        if layer.class_name == 'Dense' or is_pointwise_dense(layer):
            if layer.class_name == 'Dense':
                n_in = layer.get_attr('n_in')
                n_out = layer.get_attr('n_out')
            else:
                n_in = layer.get_attr('n_chan')
                n_out = layer.get_attr('n_filt')
            if n_in is None or n_out is None:
                raise ValueError(f'{layer.name}: missing n_in/n_out for {layer.class_name}.')
            meta['n_in'] = int(n_in)
            meta['n_out'] = int(n_out)
            meta['use_bias'] = layer.get_attr('bias_data') is not None
            meta['input_roles'] = ['lhs', 'rhs'] + (['bias'] if meta['use_bias'] else [])

        if layer.class_name == 'ApplyAlpha':
            scale = layer.get_attr('scale_data')
            if scale is not None:
                meta['scale'] = np.asarray(scale, dtype=np.float64).flatten().tolist()

        if layer.class_name == 'LayerNormalization':
            meta['input_roles'] = ['lhs', 'gamma', 'beta']
            epsilon = layer.get_attr('epsilon')
            if epsilon is not None:
                meta['epsilon'] = float(epsilon)

        # hls4ml's Softmax subclasses Activation but reports class_name='Softmax'
        # (and lowers to op_type 'softmax'); it takes the same single 'lhs' input.
        if layer.class_name in ('Activation', 'Softmax'):
            meta['input_roles'] = ['lhs']
            act = (layer.get_attr('activation', '') or '').lower()
            if act:
                meta['activation'] = act
        if layer.class_name == 'Transpose':
            meta['input_roles'] = ['lhs']
            perm = layer.get_attr('perm')
            if perm is None:
                raise ValueError(f'{layer.name}: missing Transpose perm attribute.')
            meta['perm'] = [int(x) for x in perm]
            meta['data_format'] = layer.get_attr('data_format')

        # hls4ml parses keras Add/Subtract/... into a single 'Merge' layer carrying attr op='add'/...
        # Only elementwise 'add' is wired (elementwise family); it takes two ordered inputs lhs+rhs.
        if layer.class_name == 'Merge' and (layer.get_attr('op') or '').lower() == 'add':
            meta['input_roles'] = ['lhs', 'rhs']

        meta['layer_class'] = layer.class_name
        if is_pointwise_dense(layer):
            meta['source_class'] = layer.class_name
            meta['layer_class'] = 'Dense'
        meta['source_layer'] = layer.name

        if meta:
            node.metadata.update(meta)

    def _create_context(self, model) -> AIEBackendContext:
        config = model.config
        aie_cfg = config.get_config_value('AIEConfig', {}) or {}
        part_name = aie_cfg.get('Part') or config.get_config_value('Part') or aie_cfg.get('Device') or 'unknown_part'

        catalog = load_device_catalog()
        device_entry = catalog.get(part_name, {}) or catalog.get(part_name.lower(), {})
        merged = dict(device_entry)
        merged.update(aie_cfg)
        if 'Generation' not in merged:
            merged['Generation'] = device_entry.get('Generation', '')

        device = DeviceSpec.from_config(part_name, merged)
        policies = BackendPolicies(
            fusion=config.get_config_value('AIEFusionPolicy', {}) or {},
            decomposition=config.get_config_value('AIEDecompositionPolicy', {}) or {},
            pack=config.get_config_value('AIEPackPolicy', {}) or {},
            cache=config.get_config_value('AIECachePolicy', {}) or {},
            tensors_have_batch=bool(
                (config.get_config_value('AIEFrontendPolicy', {}) or {}).get('TensorsHaveBatch', False)
            ),
        )

        project_config = ProjectConfig(
            output_dir=Path(config.get_output_dir()),
            project_name=config.get_project_name(),
            stamp=config.get_config_value('Stamp'),
            custom_sources=dict(config.backend.get_custom_source()),
        )
        ctx = AIEBackendContext(device=device, policies=policies, project_config=project_config, aie_config=aie_cfg)
        register_default_traits(ctx)
        return ctx

    def _map_op_type(self, layer) -> str:
        if layer.class_name in ('Dense',) or is_pointwise_dense(layer):
            return 'dense'
        if layer.class_name == 'Transpose':
            return 'transpose'
        if layer.class_name == 'LayerNormalization':
            return 'layer_norm'
        if layer.class_name == 'Merge':
            # keras Add/Subtract/... all become hls4ml 'Merge' with attr op; only add is supported.
            op = (layer.get_attr('op') or '').lower()
            if op == 'add':
                return 'add'
            raise NotImplementedError(
                f'{layer.name}: Merge op {op!r} is not supported (only elementwise add is wired).'
            )
        return layer.class_name.lower()

    def _normalize_transpose_perm(self, node: OpNode, output_shape) -> None:
        perm = node.metadata.get('perm')
        rank = len([int(x) for x in output_shape])
        if len(perm) == rank - 1:
            node.metadata['perm'] = [0] + [int(p) + 1 for p in perm]
