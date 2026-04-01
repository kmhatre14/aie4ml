from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...aie_types import QuantIntent
from ...ir import LogicalIR, OpNode, TensorVar
from ...ir.context import AIEBackendContext
from ...model import AIEModel
from ..common import attach_quant_role_bindings
from .utils import (
    attr,
    create_context,
    dequantize_data,
    initializer_map,
    input_maps,
    intent_from_initializer,
    intent_from_qparams,
    normalize_directives,
    require_onnx,
    resolve_project_name,
)
from .utils import (
    node_name as onnx_node_name,
)


def lower_onnx_model(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEBackendContext:
    onnx, helper, numpy_helper = require_onnx()

    model_path = None
    if isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        model_proto = onnx.load(str(model_path))
    else:
        model_proto = model_or_path

    graph_proto = model_proto.graph
    resolved_project_name = resolve_project_name(model_path, model_proto, project_name)
    ctx = create_context(config, output_dir, resolved_project_name, stamp, custom_sources)
    graph: LogicalIR = ctx.ir.logical

    batch_size = int(ctx.aie_config['BatchSize'])
    initializers = initializer_map(graph_proto, numpy_helper)
    raw_input_shapes, raw_input_dtypes = input_maps(graph_proto, helper, set(initializers))
    for name, shape in raw_input_shapes.items():
        if not shape:
            raise ValueError(f'{name}: scalar inputs are not supported.')
        if int(shape[0]) != batch_size:
            raise ValueError(
                f'{name}: input batch dimension {shape[0]} does not match AIEConfig.BatchSize={batch_size}.'
            )

    for name, shape in raw_input_shapes.items():
        tensor = TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=None)
        graph.add_tensor(tensor)
        graph.mark_graph_input(name)

    layer_directives = dict(config.get('LayerDirectives', {}) or {})
    used_directives = set()

    value_tensors: Dict[str, TensorVar] = {}
    shape_of: Dict[str, Tuple[int, ...]] = {name: tuple(arr.shape) for name, arr in initializers.items()}
    shape_of.update(raw_input_shapes)
    q_aliases: Dict[str, Tuple[str, Any, QuantIntent, Tuple[int, ...]]] = {}

    def _graph_tensor(name: str, shape: Tuple[int, ...], intent: QuantIntent) -> TensorVar:
        tensor = graph.tensors.get(name)
        if tensor is None:
            tensor = TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=intent)
            graph.add_tensor(tensor)
        elif tensor.precision is None:
            tensor.precision = intent
        elif tensor.precision != intent:
            raise ValueError(f'{name}: conflicting quantization intent for graph input.')
        return tensor

    def _param_tensor(name: str, data: np.ndarray, intent: QuantIntent) -> TensorVar:
        tensor = graph.tensors.get(name)
        shape = tuple(int(x) for x in np.asarray(data).shape)
        if tensor is None:
            tensor = TensorVar(name=name, shape=shape, precision=intent, data=np.asarray(data, dtype=np.float64))
            graph.add_tensor(tensor)
        else:
            if tensor.data is None:
                raise ValueError(f'{name}: expected parameter tensor.')
            if tensor.precision != intent:
                raise ValueError(f'{name}: conflicting quantization intent for parameter tensor.')
        return tensor

    def _source_for(name: str, node_name: str) -> TensorVar:
        if name not in value_tensors:
            raise ValueError(f'{node_name}: unsupported input {name}. Expected a dequantized activation tensor.')
        return value_tensors[name]

    def _parameter_source_for(name: str, node_name: str) -> TensorVar:
        if name in value_tensors:
            tensor = value_tensors[name]
            if not tensor.is_parameter:
                raise ValueError(f'{node_name}: parameter input {name} must be constant.')
            return tensor
        if name in initializers:
            data = np.asarray(initializers[name])
            return _param_tensor(name, data, intent_from_initializer(data, node_name))
        raise ValueError(f'{node_name}: parameter input {name} must be a constant initializer.')

    def _any_source_for(name: str, node_name: str) -> TensorVar:
        if name in value_tensors:
            return value_tensors[name]
        if name in initializers:
            return _parameter_source_for(name, node_name)
        raise ValueError(f'{node_name}: unsupported input {name}.')

    for index, node in enumerate(graph_proto.node):
        node_name = onnx_node_name(node, index)
        directives = normalize_directives(node_name, layer_directives.get(node_name))
        if directives:
            used_directives.add(node_name)

        op_type = node.op_type

        if op_type == 'QuantizeLinear':
            if len(node.input) != 3:
                raise ValueError(f'{node_name}: QuantizeLinear must have exactly 3 inputs.')
            src_name, scale_name, zero_name = node.input
            if zero_name not in initializers:
                raise ValueError(f'{node_name}: zero_point must be a constant initializer.')
            intent = intent_from_qparams(initializers, scale_name, zero_name, initializers[zero_name].dtype, node_name)
            if src_name in value_tensors:
                tensor = value_tensors[src_name]
                producer = tensor.producer
                relu_producer = (
                    producer is not None
                    and producer.op_type == 'activation'
                    and producer.metadata.get('activation') == 'relu'
                )
                if tensor.precision is None:
                    if relu_producer:
                        relu_input = producer.inputs[0] if producer.inputs else None
                        if relu_input is not None and relu_input.precision is None:
                            relu_input.precision = QuantIntent(
                                width=int(intent.width),
                                frac=int(intent.frac),
                                signed=True,
                                rounding=intent.rounding,
                                saturation=intent.saturation,
                            )
                    tensor.precision = intent
                elif tensor.precision != intent:
                    if relu_producer:
                        relu_input = producer.inputs[0] if producer.inputs else None
                        if relu_input is not None and relu_input.precision is None:
                            relu_input.precision = QuantIntent(
                                width=int(intent.width),
                                frac=int(intent.frac),
                                signed=True,
                                rounding=intent.rounding,
                                saturation=intent.saturation,
                            )
                        tensor.precision = intent
                    else:
                        raise ValueError(f'{node_name}: QuantizeLinear intent does not match source tensor precision.')
                q_aliases[node.output[0]] = ('tensor', tensor, intent, tuple(int(x) for x in tensor.shape))
                shape_of[node.output[0]] = tuple(int(x) for x in tensor.shape)
            elif src_name in raw_input_shapes:
                q_aliases[node.output[0]] = ('input', src_name, intent, raw_input_shapes[src_name])
                shape_of[node.output[0]] = raw_input_shapes[src_name]
            elif src_name in initializers:
                q_aliases[node.output[0]] = (
                    'initializer',
                    src_name,
                    intent,
                    tuple(int(x) for x in initializers[src_name].shape),
                )
                shape_of[node.output[0]] = tuple(int(x) for x in initializers[src_name].shape)
            else:
                raise ValueError(f'{node_name}: QuantizeLinear input {src_name} is unsupported.')
            continue

        if op_type == 'DequantizeLinear':
            if len(node.input) != 3:
                raise ValueError(f'{node_name}: DequantizeLinear must have exactly 3 inputs.')
            src_name, scale_name, zero_name = node.input
            out_name = node.output[0]

            if src_name in q_aliases:
                source_kind, source_ref, q_intent, src_shape = q_aliases[src_name]
                intent = intent_from_qparams(
                    initializers, scale_name, zero_name, initializers[zero_name].dtype, node_name
                )
                if intent != q_intent:
                    raise ValueError(f'{node_name}: QuantizeLinear/DequantizeLinear parameters do not match.')
                if source_kind == 'tensor':
                    tensor = source_ref
                elif source_kind == 'input':
                    tensor = _graph_tensor(source_ref, src_shape, intent)
                elif source_kind == 'initializer':
                    tensor = _param_tensor(
                        source_ref,
                        dequantize_data(initializers[source_ref], initializers, scale_name, zero_name, node_name),
                        intent,
                    )
                else:
                    raise ValueError(f'{node_name}: unsupported quantized source kind {source_kind}.')
                value_tensors[out_name] = tensor
                shape_of[out_name] = src_shape
                continue

            if src_name in initializers:
                intent = intent_from_qparams(
                    initializers, scale_name, zero_name, initializers[src_name].dtype, node_name
                )
                tensor = _param_tensor(
                    src_name,
                    dequantize_data(initializers[src_name], initializers, scale_name, zero_name, node_name),
                    intent,
                )
                value_tensors[out_name] = tensor
                shape_of[out_name] = tuple(int(x) for x in initializers[src_name].shape)
                continue

            if src_name in raw_input_shapes:
                raw_dtype = raw_input_dtypes[src_name]
                if not np.issubdtype(raw_dtype, np.integer):
                    raise ValueError(
                        f'{node_name}: direct DequantizeLinear on graph input requires integer input type.'
                    )
                intent = intent_from_qparams(initializers, scale_name, zero_name, raw_dtype, node_name)
                tensor = _graph_tensor(src_name, raw_input_shapes[src_name], intent)
                value_tensors[out_name] = tensor
                shape_of[out_name] = raw_input_shapes[src_name]
                continue

            raise ValueError(f'{node_name}: DequantizeLinear input {src_name} is unsupported.')

        if op_type == 'Transpose':
            if len(node.input) != 1:
                raise ValueError(f'{node_name}: Transpose must have exactly 1 input.')
            src = _source_for(node.input[0], node_name)
            perm = [int(x) for x in list(attr(node, 'perm'))]
            in_shape = tuple(int(x) for x in shape_of[node.input[0]])
            if sorted(perm) != list(range(len(in_shape))):
                raise ValueError(f'{node_name}: invalid permutation {perm} for rank {len(in_shape)}.')
            out_shape = tuple(in_shape[p] for p in perm)
            out_name = node.output[0]

            if src.is_parameter:
                data = np.transpose(np.asarray(src.data, dtype=np.float64), axes=perm)
                tensor = _param_tensor(out_name, data, src.precision)
                value_tensors[out_name] = tensor
                shape_of[out_name] = out_shape
                continue

            op = OpNode(name=f'{node_name}_aie', op_type='transpose', dialect=ctx.device.dialect)
            op.metadata.update(
                attach_quant_role_bindings(
                    {
                        'perm': perm,
                        'data_format': 'channels_last',
                        'layer_class': 'Transpose',
                        'source_layer': node_name,
                        'input_roles': ['lhs'],
                    }
                )
            )
            op.directives.update(directives)
            src.consumers.append(op)
            op.inputs.append(src)
            out_tensor = TensorVar(name=out_name, shape=out_shape, precision=src.precision, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[out_name] = out_tensor
            shape_of[out_name] = out_shape
            continue

        if op_type in ('MatMul', 'Gemm'):
            if op_type == 'MatMul':
                if len(node.input) != 2:
                    raise ValueError(f'{node_name}: MatMul must have exactly 2 inputs.')
                act_name, weight_name = node.input
                bias_name = None
                trans_b = False
                if attr(node, 'transA', 0) not in (0, False):
                    raise ValueError(f'{node_name}: transA is not supported.')
            else:
                if len(node.input) not in (2, 3):
                    raise ValueError(f'{node_name}: Gemm must have 2 or 3 inputs.')
                if float(attr(node, 'alpha', 1.0)) != 1.0:
                    raise ValueError(f'{node_name}: Gemm alpha must be 1.')
                if float(attr(node, 'beta', 1.0)) != 1.0:
                    raise ValueError(f'{node_name}: Gemm beta must be 1.')
                if int(attr(node, 'transA', 0)) != 0:
                    raise ValueError(f'{node_name}: Gemm transA is not supported.')
                act_name, weight_name = node.input[:2]
                bias_name = node.input[2] if len(node.input) == 3 else None
                trans_b = int(attr(node, 'transB', 0)) == 1

            act_tensor = _source_for(act_name, node_name)
            weight_tensor = _parameter_source_for(weight_name, node_name)
            if act_tensor.is_parameter:
                raise ValueError(f'{node_name}: activation input cannot be constant.')
            if not weight_tensor.is_parameter:
                raise ValueError(f'{node_name}: weight input must be a constant initializer.')

            act_shape = tuple(int(x) for x in shape_of[act_name])
            weight_data = np.asarray(weight_tensor.data, dtype=np.float64)
            if trans_b:
                weight_data = np.transpose(weight_data)
            if weight_data.ndim != 2:
                raise ValueError(f'{node_name}: only rank-2 weight matrices are supported.')
            if len(act_shape) < 1:
                raise ValueError(f'{node_name}: scalar activations are not supported.')
            if int(act_shape[-1]) != int(weight_data.shape[0]):
                raise ValueError(
                    f'{node_name}: activation feature dimension {act_shape[-1]} '
                    f'does not match weight input dimension {weight_data.shape[0]}.'
                )

            n_in = int(weight_data.shape[0])
            n_out = int(weight_data.shape[1])
            out_shape = tuple(list(act_shape[:-1]) + [n_out])

            weight_param = _param_tensor(f'{node_name}_weight', weight_data, weight_tensor.precision)
            op = OpNode(name=f'{node_name}_aie', op_type='dense', dialect=ctx.device.dialect)
            op.metadata.update(
                attach_quant_role_bindings(
                    {
                        'n_in': n_in,
                        'n_out': n_out,
                        'use_bias': False,
                        'layer_class': 'Dense',
                        'source_class': op_type,
                        'source_layer': node_name,
                        'input_roles': ['lhs', 'rhs'],
                    }
                )
            )
            op.directives.update(directives)
            act_tensor.consumers.append(op)
            op.inputs.extend([act_tensor, weight_param])

            if bias_name is not None:
                bias_tensor = _parameter_source_for(bias_name, node_name)
                if not bias_tensor.is_parameter:
                    raise ValueError(f'{node_name}: Gemm bias must be constant.')
                bias_data = np.asarray(bias_tensor.data, dtype=np.float64).reshape(-1)
                if int(bias_data.size) != n_out:
                    raise ValueError(f'{node_name}: Gemm bias must contain exactly {n_out} elements.')
                bias_param = _param_tensor(f'{node_name}_bias', bias_data, bias_tensor.precision)
                op.inputs.append(bias_param)
                op.metadata['use_bias'] = True
                op.metadata['input_roles'] = ['lhs', 'rhs', 'bias']
                attach_quant_role_bindings(op.metadata)

            out_tensor = TensorVar(name=node.output[0], shape=out_shape, precision=None, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[node.output[0]] = out_tensor
            shape_of[node.output[0]] = out_shape
            continue

        if op_type == 'Add':
            if len(node.input) != 2:
                raise ValueError(f'{node_name}: Add must have exactly 2 inputs.')
            lhs_name, rhs_name = node.input
            lhs = _any_source_for(lhs_name, node_name)
            rhs = _any_source_for(rhs_name, node_name)

            if lhs.producer is not None and lhs.producer.op_type == 'dense' and rhs.is_parameter:
                dense_tensor, bias_tensor = lhs, rhs
            elif rhs.producer is not None and rhs.producer.op_type == 'dense' and lhs.is_parameter:
                dense_tensor, bias_tensor = rhs, lhs
            else:
                raise ValueError(f'{node_name}: generic Add is not supported; only fused dense bias is supported.')

            dense_node = dense_tensor.producer
            if dense_node.metadata.get('use_bias'):
                raise ValueError(f'{node_name}: dense node already has a fused bias.')
            bias_data = np.asarray(bias_tensor.data, dtype=np.float64).reshape(-1)
            n_out = int(dense_node.metadata['n_out'])
            if int(bias_data.size) != n_out:
                raise ValueError(f'{node_name}: fused dense bias Add requires exactly {n_out} bias elements.')
            bias_param = _param_tensor(f'{node_name}_bias', bias_data, bias_tensor.precision)
            dense_node.inputs.append(bias_param)
            dense_node.metadata['use_bias'] = True
            dense_node.metadata['input_roles'] = ['lhs', 'rhs', 'bias']
            attach_quant_role_bindings(dense_node.metadata)
            value_tensors[node.output[0]] = dense_tensor
            shape_of[node.output[0]] = tuple(int(x) for x in dense_tensor.shape)
            continue

        if op_type == 'Relu':
            if len(node.input) != 1:
                raise ValueError(f'{node_name}: Relu must have exactly 1 input.')
            src = _source_for(node.input[0], node_name)
            out_name = node.output[0]
            out_shape = tuple(int(x) for x in shape_of[node.input[0]])
            op = OpNode(name=f'{node_name}_aie', op_type='activation', dialect=ctx.device.dialect)
            op.metadata.update(
                attach_quant_role_bindings(
                    {
                        'activation': 'relu',
                        'layer_class': 'Activation',
                        'source_layer': node_name,
                        'input_roles': ['lhs'],
                    }
                )
            )
            op.directives.update(directives)
            src.consumers.append(op)
            op.inputs.append(src)
            out_tensor = TensorVar(name=out_name, shape=out_shape, precision=src.precision, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[out_name] = out_tensor
            shape_of[out_name] = out_shape
            continue

        raise ValueError(f'{node_name}: unsupported ONNX op {op_type}.')

    unused_directives = sorted(set(layer_directives) - used_directives)
    if unused_directives:
        raise ValueError('Unused LayerDirectives entries: ' + ', '.join(unused_directives))

    for output in graph_proto.output:
        if output.name not in value_tensors:
            raise ValueError(f'Graph output {output.name}: expected a lowered semantic tensor.')
        tensor = value_tensors[output.name]
        if output.name not in graph.tensors:
            if tensor.name in graph.tensors:
                graph.tensors.pop(tensor.name)
            tensor.name = output.name
            graph.tensors[output.name] = tensor
        graph.mark_graph_output(output.name)

    for tensor in graph.tensors.values():
        if tensor.is_parameter:
            continue
        if tensor.precision is None:
            raise ValueError(f'{tensor.name}: missing quantization intent after ONNX QDQ lowering.')

    return ctx


def from_onnx(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEModel:
    ctx = lower_onnx_model(
        model_or_path,
        config,
        output_dir=output_dir,
        project_name=project_name,
        stamp=stamp,
        custom_sources=custom_sources,
    )
    return AIEModel.from_context(ctx, source_model=model_or_path)
