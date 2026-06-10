from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...aie_types import FloatFormat, FloatIntent, QuantIntent
from ...ir import LogicalIR, OpNode, TensorVar, set_input_roles
from ...ir.context import AIEBackendContext
from ...model import AIEModel
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

_ONNX_FLOAT = 1
_ONNX_BFLOAT16 = 16
_ONNX_FP8E4M3FN = 17
_ONNX_INT_ELEM_TYPE_TO_NP = {2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64}


def _normalize_axis(axis: int, rank: int, node_name: str, op_type: str) -> int:
    if rank < 1:
        raise ValueError(f'{node_name}: {op_type} does not accept scalar tensors.')
    normalized = int(axis)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0 or normalized >= int(rank):
        raise ValueError(f'{node_name}: {op_type} axis {axis} is out of range for rank {rank}.')
    return normalized


def _broadcast_prefix(lhs: Tuple[int, ...], rhs: Tuple[int, ...], node_name: str) -> Tuple[int, ...]:
    out = []
    lhs_rev = list(reversed(lhs))
    rhs_rev = list(reversed(rhs))
    for index in range(max(len(lhs_rev), len(rhs_rev))):
        lhs_dim = lhs_rev[index] if index < len(lhs_rev) else 1
        rhs_dim = rhs_rev[index] if index < len(rhs_rev) else 1
        if lhs_dim == 1:
            out.append(int(rhs_dim))
        elif rhs_dim == 1 or lhs_dim == rhs_dim:
            out.append(int(lhs_dim))
        else:
            raise ValueError(f'{node_name}: ONNX MatMul leading dimensions are not broadcastable: {lhs} and {rhs}.')
    return tuple(reversed(out))


def _matmul_output_shape(lhs_shape: Tuple[int, ...], rhs_shape: Tuple[int, ...], node_name: str) -> Tuple[int, ...]:
    """Apply ONNX MatMul shape semantics only; implementation legality is resolver-owned."""
    if len(lhs_shape) < 1 or len(rhs_shape) < 1:
        raise ValueError(f'{node_name}: ONNX MatMul does not accept scalar inputs.')

    lhs_is_vector = len(lhs_shape) == 1
    rhs_is_vector = len(rhs_shape) == 1

    lhs_k = int(lhs_shape[-1])
    rhs_k = int(rhs_shape[0] if rhs_is_vector else rhs_shape[-2])
    if lhs_k != rhs_k:
        raise ValueError(f'{node_name}: ONNX MatMul K dimensions do not match: {lhs_k} and {rhs_k}.')

    lhs_prefix = () if lhs_is_vector else tuple(int(x) for x in lhs_shape[:-2])
    rhs_prefix = () if rhs_is_vector else tuple(int(x) for x in rhs_shape[:-2])
    out = list(_broadcast_prefix(lhs_prefix, rhs_prefix, node_name))
    if not lhs_is_vector:
        out.append(int(lhs_shape[-2]))
    if not rhs_is_vector:
        out.append(int(rhs_shape[-1]))
    return tuple(out)


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

    initializers = initializer_map(graph_proto, numpy_helper)
    raw_input_shapes, raw_input_dtypes = input_maps(graph_proto, set(initializers))
    for name, shape in raw_input_shapes.items():
        if not shape:
            raise ValueError(f'{name}: scalar inputs are not supported.')

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
            if name in raw_input_shapes:
                tensor = _any_source_for(name, node_name)
                if tensor.is_parameter:
                    raise ValueError(f'{node_name}: activation input {name} cannot be constant.')
                return tensor
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
        # Direct float graph input (bfloat16 / float32) — no QDQ wrapper.
        if name in raw_input_shapes:
            elem_type = raw_input_dtypes[name]
            if elem_type == _ONNX_FLOAT:
                intent = FloatIntent(width=32, format=FloatFormat.FP32)
            elif elem_type == _ONNX_BFLOAT16:
                intent = FloatIntent(width=16, format=FloatFormat.BF16)
            elif elem_type == _ONNX_FP8E4M3FN:
                intent = FloatIntent(width=8, format=FloatFormat.FP8_E4M3)
            else:
                raise ValueError(
                    f'{node_name}: graph input "{name}" has unsupported ONNX elem_type {elem_type}. '
                    'Use QDQ wrapping for integer inputs.'
                )
            tensor = _graph_tensor(name, raw_input_shapes[name], intent)
            value_tensors[name] = tensor
            return tensor
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
                scale_producer = producer is not None and producer.op_type == 'scale'
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
                    elif scale_producer:
                        scale_input = producer.inputs[0] if producer.inputs else None
                        if scale_input is not None and scale_input.precision is None:
                            scale_input.precision = intent
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
                raw_elem_type = raw_input_dtypes[src_name]
                raw_np_dtype = _ONNX_INT_ELEM_TYPE_TO_NP.get(raw_elem_type)
                if raw_np_dtype is None:
                    raise ValueError(
                        f'{node_name}: direct DequantizeLinear on graph input requires integer input type.'
                    )
                intent = intent_from_qparams(initializers, scale_name, zero_name, raw_np_dtype, node_name)
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
                {
                    'perm': perm,
                    'data_format': 'channels_last',
                    'layer_class': 'Transpose',
                    'source_layer': node_name,
                    'input_roles': ['lhs'],
                }
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

        if op_type in ('Slice', 'Split'):
            if op_type == 'Slice':
                if len(node.input) not in (3, 4, 5):
                    raise ValueError(f'{node_name}: Slice must have 3 to 5 inputs.')
                src_name = node.input[0]
                for name in node.input[1:]:
                    if name and name not in initializers:
                        raise NotImplementedError(f'{node_name}: Slice parameters must be constant initializers.')
                starts = np.asarray(initializers[node.input[1]], dtype=np.int64).reshape(-1)
                ends = np.asarray(initializers[node.input[2]], dtype=np.int64).reshape(-1)
                axes = (
                    np.asarray(initializers[node.input[3]], dtype=np.int64).reshape(-1)
                    if len(node.input) >= 4 and node.input[3]
                    else np.arange(starts.size, dtype=np.int64)
                )
                steps = (
                    np.asarray(initializers[node.input[4]], dtype=np.int64).reshape(-1)
                    if len(node.input) >= 5 and node.input[4]
                    else np.ones(starts.size, dtype=np.int64)
                )
                if starts.size != 1 or ends.size != 1 or axes.size != 1 or steps.size != 1:
                    raise NotImplementedError(f'{node_name}: Slice currently supports exactly one sliced axis.')
                if int(steps[0]) != 1:
                    raise NotImplementedError(f'{node_name}: Slice currently supports only unit steps.')
                src_shape = tuple(int(x) for x in shape_of[src_name])
                axis = _normalize_axis(int(axes[0]), len(src_shape), node_name, 'Slice')
                start = max(0, min(int(src_shape[axis]), int(starts[0])))
                end = max(start, min(int(src_shape[axis]), int(ends[0])))
                ranges = [(start, end - start)]
            else:
                if len(node.input) not in (1, 2):
                    raise ValueError(f'{node_name}: Split must have 1 or 2 inputs.')
                src_name = node.input[0]
                src_shape = tuple(int(x) for x in shape_of[src_name])
                axis = _normalize_axis(int(attr(node, 'axis', 0)), len(src_shape), node_name, 'Split')
                if len(node.input) == 2:
                    split_name = node.input[1]
                    if split_name not in initializers:
                        raise NotImplementedError(f'{node_name}: Split sizes must be a constant initializer.')
                    sizes = [int(x) for x in np.asarray(initializers[split_name], dtype=np.int64).reshape(-1)]
                else:
                    sizes = [int(x) for x in list(attr(node, 'split', []))]
                    if not sizes:
                        if int(src_shape[axis]) % len(node.output) != 0:
                            raise ValueError(f'{node_name}: equal Split does not divide axis {axis} exactly.')
                        sizes = [int(src_shape[axis]) // len(node.output) for _ in node.output]
                if len(sizes) != len(node.output) or any(size <= 0 for size in sizes):
                    raise ValueError(f'{node_name}: Split sizes must be positive and match output count.')
                if sum(sizes) != int(src_shape[axis]):
                    raise ValueError(f'{node_name}: Split sizes must cover axis {axis} exactly.')
                offset = 0
                ranges = []
                for size in sizes:
                    ranges.append((offset, size))
                    offset += size

            source = _source_for(src_name, node_name)
            op = OpNode(name=f'{node_name}_aie', op_type=op_type.lower(), dialect=ctx.device.dialect)
            op.metadata.update(
                {
                    'axis': axis,
                    'slices': [{'start': start, 'extent': extent} for start, extent in ranges],
                    'layer_class': op_type,
                    'source_class': op_type,
                    'source_layer': node_name,
                    'input_roles': ['lhs'],
                }
            )
            source.consumers.append(op)
            op.inputs.append(source)
            for out_name, (start, extent) in zip(node.output, ranges):
                out_shape = list(src_shape)
                out_shape[axis] = extent
                out_tensor = TensorVar(
                    name=out_name,
                    shape=tuple(out_shape),
                    precision=source.precision,
                    producer=op,
                )
                graph.add_tensor(out_tensor)
                op.outputs.append(out_tensor)
                value_tensors[out_name] = out_tensor
                shape_of[out_name] = tuple(out_shape)
            graph.add_node(op)
            continue

        if op_type in ('MatMul', 'Gemm'):
            if op_type == 'MatMul':
                if len(node.input) != 2:
                    raise ValueError(f'{node_name}: MatMul must have exactly 2 inputs.')
                lhs_name, rhs_name = node.input
                bias_name = None
                trans_b = False
                if attr(node, 'transA', 0) not in (0, False):
                    raise ValueError(f'{node_name}: transA is not supported.')
                rhs_is_constant = rhs_name in initializers or (
                    rhs_name in value_tensors and value_tensors[rhs_name].is_parameter
                )
            else:
                if len(node.input) not in (2, 3):
                    raise ValueError(f'{node_name}: Gemm must have 2 or 3 inputs.')
                if float(attr(node, 'alpha', 1.0)) != 1.0:
                    raise ValueError(f'{node_name}: Gemm alpha must be 1.')
                if float(attr(node, 'beta', 1.0)) != 1.0:
                    raise ValueError(f'{node_name}: Gemm beta must be 1.')
                if int(attr(node, 'transA', 0)) != 0:
                    raise ValueError(f'{node_name}: Gemm transA is not supported.')
                lhs_name, rhs_name = node.input[:2]
                bias_name = node.input[2] if len(node.input) == 3 else None
                trans_b = int(attr(node, 'transB', 0)) == 1
                # NOTE: Gemm is currently normalized only through the static-RHS dense path.
                # Dynamic RHS Gemm is intentionally unsupported here; add an explicit
                # Gemm -> MatMul(+Add) canonicalization first if that subset is needed.
                rhs_is_constant = True

            lhs_tensor = _source_for(lhs_name, node_name)
            if lhs_tensor.is_parameter:
                raise ValueError(f'{node_name}: activation input cannot be constant.')

            lhs_shape = tuple(int(x) for x in shape_of[lhs_name])

            if rhs_is_constant:
                rhs_tensor = _parameter_source_for(rhs_name, node_name)
                if not rhs_tensor.is_parameter:
                    raise ValueError(f'{node_name}: weight input must be a constant initializer.')

                rhs_data = np.asarray(rhs_tensor.data, dtype=np.float64)
                if trans_b:
                    rhs_data = np.transpose(rhs_data)
                if rhs_data.ndim != 2:
                    if op_type == 'Gemm':
                        raise ValueError(f'{node_name}: ONNX Gemm weight matrix must be rank-2.')
                    out_shape = _matmul_output_shape(lhs_shape, tuple(int(x) for x in rhs_data.shape), node_name)
                    n_in = int(rhs_data.shape[0] if rhs_data.ndim == 1 else rhs_data.shape[-2])
                    n_out = int(1 if rhs_data.ndim == 1 else rhs_data.shape[-1])
                    rhs_param = _param_tensor(f'{node_name}_rhs', rhs_data, rhs_tensor.precision)
                    op = OpNode(name=f'{node_name}_aie', op_type='matmul', dialect=ctx.device.dialect)
                    op.metadata.update(
                        {
                            'n_in': n_in,
                            'n_out': n_out,
                            'use_bias': False,
                            'layer_class': 'MatMul',
                            'source_class': 'MatMul',
                            'source_layer': node_name,
                            'input_roles': ['lhs', 'rhs'],
                        }
                    )
                    op.directives.update(directives)
                    lhs_tensor.consumers.append(op)
                    op.inputs.extend([lhs_tensor, rhs_param])
                    out_precision = lhs_tensor.precision if isinstance(lhs_tensor.precision, FloatIntent) else None
                    out_tensor = TensorVar(name=node.output[0], shape=out_shape, precision=out_precision, producer=op)
                    graph.add_tensor(out_tensor)
                    op.outputs.append(out_tensor)
                    graph.add_node(op)
                    value_tensors[node.output[0]] = out_tensor
                    shape_of[node.output[0]] = out_shape
                    continue
                out_shape = _matmul_output_shape(lhs_shape, tuple(int(x) for x in rhs_data.shape), node_name)
                n_in = int(rhs_data.shape[0])
                n_out = int(rhs_data.shape[1])

                rhs_param = _param_tensor(f'{node_name}_weight', rhs_data, rhs_tensor.precision)
                op = OpNode(name=f'{node_name}_aie', op_type='dense', dialect=ctx.device.dialect)
                op.metadata.update(
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

                op.directives.update(directives)
                lhs_tensor.consumers.append(op)
                op.inputs.extend([lhs_tensor, rhs_param])

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
            else:
                if bias_name is not None:
                    raise ValueError(f'{node_name}: dynamic MatMul does not support fused bias.')
                rhs_tensor = _source_for(rhs_name, node_name)
                rhs_shape = tuple(int(x) for x in shape_of[rhs_name])
                out_shape = _matmul_output_shape(lhs_shape, rhs_shape, node_name)
                n_in = int(rhs_shape[0] if len(rhs_shape) == 1 else rhs_shape[-2])
                n_out = int(1 if len(rhs_shape) == 1 else rhs_shape[-1])

                op = OpNode(name=f'{node_name}_aie', op_type='matmul', dialect=ctx.device.dialect)
                op.metadata.update(
                    {
                        'n_in': n_in,
                        'n_out': n_out,
                        'use_bias': False,
                        'layer_class': 'MatMul',
                        'source_class': 'MatMul',
                        'source_layer': node_name,
                        'input_roles': ['lhs', 'rhs'],
                    }
                )

                op.directives.update(directives)
                lhs_tensor.consumers.append(op)
                rhs_tensor.consumers.append(op)
                op.inputs.extend([lhs_tensor, rhs_tensor])

            out_precision = lhs_tensor.precision if isinstance(lhs_tensor.precision, FloatIntent) else None
            out_tensor = TensorVar(name=node.output[0], shape=out_shape, precision=out_precision, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[node.output[0]] = out_tensor
            shape_of[node.output[0]] = out_shape
            continue

        if op_type in ('Mul', 'Div'):
            if len(node.input) != 2:
                raise ValueError(f'{node_name}: {op_type} must have exactly 2 inputs.')

            if op_type == 'Mul':
                constant_names = [name for name in node.input if name in initializers]
                if len(constant_names) != 1:
                    raise NotImplementedError(f'{node_name}: Mul currently requires exactly one constant input.')
                constant_name = constant_names[0]
                source_name = node.input[1] if node.input[0] == constant_name else node.input[0]
                scale = float(np.asarray(initializers[constant_name]).reshape(-1)[0])
            else:
                source_name, constant_name = node.input
                if constant_name not in initializers:
                    raise NotImplementedError(f'{node_name}: Div currently requires a constant divisor.')
                divisor = float(np.asarray(initializers[constant_name]).reshape(-1)[0])
                if divisor == 0.0:
                    raise ValueError(f'{node_name}: Div constant divisor must be nonzero.')
                scale = 1.0 / divisor

            constant = np.asarray(initializers[constant_name])
            if constant.size != 1:
                raise NotImplementedError(f'{node_name}: {op_type} currently requires a scalar constant.')

            source = _source_for(source_name, node_name)
            op = OpNode(name=f'{node_name}_aie', op_type='scale', dialect=ctx.device.dialect)
            op.metadata.update(
                {
                    'scale': scale,
                    'layer_class': op_type,
                    'source_class': op_type,
                    'source_layer': node_name,
                    'input_roles': ['lhs'],
                }
            )
            op.directives.update(directives)
            source.consumers.append(op)
            op.inputs.append(source)
            out_tensor = TensorVar(
                name=node.output[0],
                shape=tuple(int(x) for x in source.shape),
                precision=source.precision,
                producer=op,
            )
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[node.output[0]] = out_tensor
            shape_of[node.output[0]] = tuple(int(x) for x in source.shape)
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
                lhs_shape = tuple(int(x) for x in shape_of[lhs_name])
                rhs_shape = tuple(int(x) for x in shape_of[rhs_name])
                if lhs_shape != rhs_shape:
                    raise ValueError(
                        f'{node_name}: generic Add only supports exact-shape elementwise inputs; '
                        f'got {lhs_shape} and {rhs_shape}.'
                    )
                op = OpNode(name=f'{node_name}_aie', op_type='add', dialect=ctx.device.dialect)
                op.metadata.update(
                    {
                        'layer_class': 'Add',
                        'source_class': 'Add',
                        'source_layer': node_name,
                        'input_roles': ['lhs', 'rhs'],
                    }
                )

                op.directives.update(directives)
                lhs.consumers.append(op)
                rhs.consumers.append(op)
                op.inputs.extend([lhs, rhs])

                # For float (bfloat16/float32) graphs, propagate precision from lhs.
                out_precision = lhs.precision if isinstance(lhs.precision, FloatIntent) else None
                out_tensor = TensorVar(name=node.output[0], shape=lhs_shape, precision=out_precision, producer=op)
                graph.add_tensor(out_tensor)
                op.outputs.append(out_tensor)
                graph.add_node(op)
                value_tensors[node.output[0]] = out_tensor
                shape_of[node.output[0]] = lhs_shape
                continue

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
            value_tensors[node.output[0]] = dense_tensor
            shape_of[node.output[0]] = tuple(int(x) for x in dense_tensor.shape)
            continue

        if op_type == 'Concat':
            if len(node.input) < 1:
                raise ValueError(f'{node_name}: Concat must have at least one input.')
            sources = [_source_for(name, node_name) for name in node.input]
            if any(src.is_parameter for src in sources):
                raise ValueError(f'{node_name}: Concat currently supports activation tensors only.')

            shapes = [tuple(int(x) for x in shape_of[name]) for name in node.input]
            rank = len(shapes[0])
            if rank < 1:
                raise ValueError(f'{node_name}: Concat does not accept scalar inputs.')
            if any(len(shape) != rank for shape in shapes):
                raise ValueError(f'{node_name}: Concat inputs must have the same rank.')
            axis = _normalize_axis(int(attr(node, 'axis', -1)), rank, node_name, 'Concat')

            prefix = tuple(shapes[0][:axis])
            suffix = tuple(shapes[0][axis + 1 :])
            for shape in shapes[1:]:
                if tuple(shape[:axis]) != prefix or tuple(shape[axis + 1 :]) != suffix:
                    raise ValueError(f'{node_name}: Concat input shapes disagree outside axis {axis}: {shapes}.')

            precision = sources[0].precision
            for src in sources[1:]:
                if src.precision != precision:
                    raise ValueError(f'{node_name}: Concat inputs must use identical precision contracts.')

            axis_extent = sum(int(shape[axis]) for shape in shapes)

            out_shape = tuple(prefix + (int(axis_extent),) + suffix)
            op = OpNode(name=f'{node_name}_aie', op_type='concat', dialect=ctx.device.dialect)
            op.metadata.update(
                {
                    'axis': axis,
                    'layer_class': 'Concat',
                    'source_class': 'Concat',
                    'source_layer': node_name,
                }
            )

            for src in sources:
                src.consumers.append(op)
            op.inputs.extend(sources)
            out_tensor = TensorVar(name=node.output[0], shape=out_shape, precision=precision, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[node.output[0]] = out_tensor
            shape_of[node.output[0]] = out_shape
            continue

        if op_type == 'LayerNormalization':
            if len(node.input) not in (2, 3):
                raise ValueError(f'{node_name}: LayerNormalization must have 2 or 3 inputs.')
            x_name = node.input[0]
            scale_name = node.input[1]
            bias_name = node.input[2] if len(node.input) == 3 else None
            x_shape = tuple(int(x) for x in shape_of[x_name])

            axis = _normalize_axis(int(attr(node, 'axis', -1)), len(x_shape), node_name, 'LayerNormalization')
            norm_shape = tuple(int(x) for x in x_shape[axis:])
            epsilon = float(attr(node, 'epsilon', 1e-5))

            x_tensor = _source_for(x_name, node_name)
            scale_tensor = _parameter_source_for(scale_name, node_name)
            if not scale_tensor.is_parameter:
                raise ValueError(f'{node_name}: LayerNormalization Scale must be a constant initializer.')

            if bias_name is not None:
                bias_tensor = _parameter_source_for(bias_name, node_name)
                if not bias_tensor.is_parameter:
                    raise ValueError(f'{node_name}: LayerNormalization Bias must be a constant initializer.')
            else:
                zeros = np.zeros(norm_shape, dtype=np.float64)
                bias_tensor = _param_tensor(f'{node_name}_beta_zero', zeros, scale_tensor.precision)

            op = OpNode(name=f'{node_name}_aie', op_type='layer_norm', dialect=ctx.device.dialect)
            inputs = [x_tensor, scale_tensor, bias_tensor]

            op.metadata.update(
                {
                    'layer_class': 'LayerNormalization',
                    'source_class': 'LayerNormalization',
                    'source_layer': node_name,
                    'input_roles': ['lhs', 'gamma', 'beta'],
                    'epsilon': epsilon,
                    'axis': axis,
                }
            )

            op.directives.update(directives)
            x_tensor.consumers.append(op)
            op.inputs.extend(inputs)

            out_precision = x_tensor.precision if isinstance(x_tensor.precision, FloatIntent) else None
            out_tensor = TensorVar(name=node.output[0], shape=x_shape, precision=out_precision, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[node.output[0]] = out_tensor
            shape_of[node.output[0]] = x_shape
            continue

        if op_type == 'Softmax':
            if len(node.input) != 1:
                raise ValueError(f'{node_name}: Softmax must have exactly 1 input.')
            if 'hccs' not in directives:
                raise ValueError(
                    f'{node_name}: ONNX Softmax lowering requires explicit HCCS directives; '
                    'this is a calibrated surrogate, not normal exponential softmax.'
                )
            src = _source_for(node.input[0], node_name)
            out_name = node.output[0]
            in_shape = tuple(int(x) for x in shape_of[node.input[0]])
            axis = _normalize_axis(int(attr(node, 'axis', -1)), len(in_shape), node_name, 'Softmax')

            op = OpNode(name=f'{node_name}_aie', op_type='softmax', dialect=ctx.device.dialect)
            op.metadata.update(
                {
                    'axis': axis,
                    'layer_class': 'Softmax',
                    'source_class': 'Softmax',
                    'source_layer': node_name,
                    'input_roles': ['lhs'],
                }
            )

            op.directives.update(directives)
            src.consumers.append(op)
            op.inputs.append(src)
            out_tensor = TensorVar(name=out_name, shape=in_shape, precision=None, producer=op)
            graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            graph.add_node(op)
            value_tensors[out_name] = out_tensor
            shape_of[out_name] = in_shape
            continue

        if op_type == 'Relu':
            if len(node.input) != 1:
                raise ValueError(f'{node_name}: Relu must have exactly 1 input.')
            src = _source_for(node.input[0], node_name)
            out_name = node.output[0]
            out_shape = tuple(int(x) for x in shape_of[node.input[0]])
            op = OpNode(name=f'{node_name}_aie', op_type='activation', dialect=ctx.device.dialect)
            op.metadata.update(
                {
                    'activation': 'relu',
                    'layer_class': 'Activation',
                    'source_layer': node_name,
                    'input_roles': ['lhs'],
                }
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

    for node in graph.nodes:
        role_names = list(node.metadata.get('input_roles') or [])
        if role_names:
            set_input_roles(node, node.inputs, role_names)

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
