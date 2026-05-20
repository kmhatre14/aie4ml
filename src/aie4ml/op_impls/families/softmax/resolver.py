from __future__ import annotations

import math

from ....aie_types import AIEDataType, FloatIntent
from ....ir import input_tensor_for_role
from ...family_registry import family_resolver
from ...registry import select_variant as _select_variant
from ...utils import build_tensor_view
from ...utils.io import view_shape
from ...utils.precision import resolve_exact_storage_dtype, storage_bytes_for_spec
from .common import DEFAULT_INV_SHIFT, infer_hccs_param_sets, resolve_softmax_parallelism, softmax_vec_size
from .config import SoftmaxConfig, SoftmaxParallelismConfig


def _hccs_directives(node_name: str, directives) -> dict:
    directives = directives or {}
    approximation = str(directives.get('approximation', 'hccs')).lower()
    if approximation != 'hccs':
        raise ValueError(f'{node_name}: only approximation="hccs" is supported for integer Softmax.')
    hccs = dict(directives.get('hccs', {}) or {})
    missing = [name for name in ('B', 'S', 'Dmax') if name not in hccs]
    if missing:
        raise ValueError(f'{node_name}: HCCS Softmax directives missing {", ".join(missing)}.')
    return hccs


@family_resolver('softmax')
class SoftmaxResolver:
    op_type = 'softmax'

    def resolve(self, node, device, directives=None) -> SoftmaxConfig:
        io_route = dict((directives or {}).get('io_route', {}))
        parallel_cfg = (directives or {}).get('parallelism', {}) or {}
        hccs = _hccs_directives(node.name, directives)

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]
        if isinstance(in_tensor.precision, FloatIntent) or isinstance(out_tensor.precision, FloatIntent):
            raise ValueError(f'{node.name}: HCCS Softmax requires integer input/output precision.')

        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'B': AIEDataType(format='int16'),
            'S': AIEDataType(format='int8'),
            'Dmax': AIEDataType(format='uint8'),
        }
        if precision['lhs'].format != 'int8' or precision['output'].format not in ('uint8', 'int16'):
            raise ValueError(
                f'{node.name}: HCCS Softmax requires signed int8 input and uint8 or int16 output, '
                f"got input={precision['lhs'].format!r}, output={precision['output'].format!r}."
            )

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        out_shape = tuple(int(x) for x in view_shape(node, out_tensor, 'outputs'))
        if in_shape != out_shape:
            raise ValueError(f'{node.name}: Softmax input/output shapes must match, got {in_shape} and {out_shape}.')
        if len(in_shape) < 2:
            raise ValueError(f'{node.name}: Softmax requires rank>=2 tensors, got {len(in_shape)}.')

        axis = int(node.metadata.get('axis', -1))
        if axis < 0:
            axis += len(in_shape)
        if axis != len(in_shape) - 1:
            raise ValueError(f'{node.name}: only last-axis Softmax is supported; got axis={axis}.')

        full_inner = int(in_shape[-1])
        outer_prefix = int(math.prod(in_shape[:-2])) if len(in_shape) > 2 else 1
        full_outer = int(in_shape[-2])

        vec_size = softmax_vec_size(precision['lhs'], device)
        if full_inner % vec_size != 0:
            raise ValueError(
                f'{node.name}: softmax axis length {full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the softmax dimension before lowering.'
            )

        cas_length = int(parallel_cfg.get('cas_length', 1))
        if cas_length != 1:
            raise ValueError(f'{node.name}: HCCS Softmax requires cas_length=1, got {cas_length}.')

        requested_cas_num = parallel_cfg.get('cas_num')
        tiling = resolve_softmax_parallelism(
            outer_prefix=outer_prefix,
            last_outer=full_outer,
            full_inner=full_inner,
            elem_in_bytes=storage_bytes_for_spec(precision['lhs']),
            elem_out_bytes=storage_bytes_for_spec(precision['output']),
            device=device,
            requested_cas_num=int(requested_cas_num) if requested_cas_num is not None else None,
        )

        io_views = {}
        io_views[in_tensor.name] = build_tensor_view(
            node,
            in_tensor,
            'inputs',
            full_inner=full_inner,
            tile_inner=full_inner,
            tile_inner_raw=full_inner,
            full_outer=full_outer,
            tile_outer=tiling.tile_outer,
            tile_outer_raw=tiling.tile_outer,
        )
        io_views[out_tensor.name] = build_tensor_view(
            node,
            out_tensor,
            'outputs',
            full_inner=full_inner,
            tile_inner=full_inner,
            tile_inner_raw=full_inner,
            full_outer=full_outer,
            tile_outer=tiling.tile_outer,
            tile_outer_raw=tiling.tile_outer,
        )

        param_sets = infer_hccs_param_sets(hccs)
        if param_sets != 1:
            raise ValueError(
                f'{node.name}: fused multi-head HCCS Softmax with param_sets={param_sets} is not supported yet. '
                'Lower attention to one Softmax op per head so each kernel uses one static HCCS parameter set.'
            )

        return SoftmaxConfig(
            precision=precision,
            parallelism=SoftmaxParallelismConfig(cas_num=int(tiling.cas_num), cas_length=cas_length),
            param_sets=int(param_sets),
            vec_size=int(vec_size),
            inv_shift=int(hccs.get('inv_shift', DEFAULT_INV_SHIFT)),
            use_clb=bool(hccs.get('use_clb', False)),
            io_views=io_views,
            io_route=io_route,
            hccs=hccs,
        )

    def select_variant(self, config: SoftmaxConfig, generation: str):
        return _select_variant(self.op_type, config, generation)
