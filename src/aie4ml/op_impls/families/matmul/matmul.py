from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, input_role, input_tensor_for_role
from ...base import OpImplConfig, OpImplFootprint, OpImplPlacementContext, OpImplSelectionContext
from ...common_types import PortBinding, PortMap
from .common import TILING_OPTIONS, select_generation_key, validate_family_tile_contract
from .dense import DenseOpImplVariant
from .types import MatmulParallelismConfig, MatmulTilingConfig


@dataclass(frozen=True)
class MatmulFlags:
    """Matmul implementation behaviour flags used by templates and codegen."""

    transpose_lhs: bool
    transpose_rhs: bool


@dataclass(frozen=True)
class MatmulOpImplParameters:
    """Matmul-specific compile-time parameters for the selected AIE implementation."""

    precision: Dict[str, Any]
    parallelism: MatmulParallelismConfig
    tiling: MatmulTilingConfig
    flags: MatmulFlags
    lhs_feat_slice: int
    rhs_feat_slice: int
    lhs_slice_raw: int
    rhs_slice_raw: int
    padded_independent_extent: int
    padded_lhs_features: int
    padded_rhs_features: int
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    io_shapes: Dict[str, Any]


class MatmulOpImplVariant(DenseOpImplVariant):
    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        tensor = next((value for value in node.inputs if value.name == tensor_name), None)
        if tensor is None:
            raise ValueError(f'{node.name}: unknown input tensor {tensor_name}.')
        role = input_role(node, tensor_name)
        if role == 'rhs':
            return self._describe_matmul_rhs(node, config, tensor_name, port, buf_dims)
        return self._describe_dense_ifm(node, config, tensor_name, port, buf_dims)

    def build_config(self, context: OpImplSelectionContext) -> OpImplConfig:
        attrs = context.attributes
        parallel = dict(attrs.parallelism)
        tiling = dict(attrs.tiling)
        slices = dict(attrs.slices)
        scalars = dict(attrs.scalars)
        lhs_tensor = input_tensor_for_role(context.node, 'lhs')
        rhs_tensor = input_tensor_for_role(context.node, 'rhs')
        lhs_view = self._io_view(context.node, lhs_tensor.name, 'inputs')
        rhs_view = self._io_view(context.node, rhs_tensor.name, 'inputs')
        perm = lhs_view.get('perm')
        transpose_lhs = perm is not None and perm[-1] != (len(perm) - 1)
        rhs_perm = rhs_view.get('perm')
        transpose_rhs = rhs_perm is not None and rhs_perm[-1] != (len(rhs_perm) - 1)
        cas = int(parallel['cas_length'])
        chains = int(parallel['cas_num'])

        params = MatmulOpImplParameters(
            precision={
                key: attrs.numeric.get(key)
                for key in ('lhs', 'rhs', 'output', 'acc')
                if attrs.numeric.get(key) is not None
            },
            parallelism=MatmulParallelismConfig(cas_length=cas, cas_num=chains),
            tiling=MatmulTilingConfig(
                tile_m=int(tiling['tile_m']),
                tile_k=int(tiling['tile_k']),
                tile_n=int(tiling['tile_n']),
            ),
            flags=MatmulFlags(transpose_lhs=bool(transpose_lhs), transpose_rhs=bool(transpose_rhs)),
            lhs_feat_slice=int(slices['lhs']),
            rhs_feat_slice=int(slices['rhs']),
            lhs_slice_raw=int(slices['lhs_raw']),
            rhs_slice_raw=int(slices['rhs_raw']),
            padded_independent_extent=int(scalars['padded_independent_extent']),
            padded_lhs_features=int(scalars['padded_lhs_features']),
            padded_rhs_features=int(scalars['padded_rhs_features']),
            shift=int(scalars['shift']),
            accumulator_tag=scalars.get('accumulator_tag'),
            rounding_mode=scalars.get('rounding_mode'),
            io_shapes=scalars['io_shapes'],
        )

        ports = PortMap(
            inputs={
                lhs_tensor.name: PortBinding(group='inA', count=cas),
                rhs_tensor.name: PortBinding(group='inB', count=cas * chains),
            },
            outputs={context.node.outputs[0].name: PortBinding(group='outC', count=chains)},
        )

        return OpImplConfig(
            variant_id=self.variant_id,
            param_template='matmul',
            graph_header='matmul_graph.h',
            graph_name='matmul_graph',
            parameters=params,
            ports=ports,
            io_route=dict(attrs.io_route),
        )

    def tiling_options(self, generation: str, query):
        return list(TILING_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def validate_config(self, context: OpImplSelectionContext, config: OpImplConfig) -> None:
        p = config.parameters
        # NOTE: this is a second-line invariant check for the current AIE-ML bank model.
        # Search-time legality already uses device.bank_mem_bytes in the resolver.
        # If a future device family exposes a different bank size here, update this guard
        # together with the device model / validation context.
        bank_bytes = 16 * 1024
        stack_bytes = 1024

        rhs_prec = p.precision['rhs']
        if not isinstance(context.node.inputs[1].precision, FloatIntent) and not bool(rhs_prec.signed):
            raise ValueError(f'{context.node.name}: matmul RHS must use a signed integer precision.')

        rhs_shape = tuple(int(x) for x in context.node.inputs[1].shape)
        if len(rhs_shape) < 2:
            raise ValueError(f'{context.node.name}: matmul RHS must be rank >=2, got {len(rhs_shape)}.')
        validate_family_tile_contract(
            node_name=context.node.name,
            precision=p.precision,
            parallelism=p.parallelism,
            tiling=p.tiling,
            padded_independent_extent=p.padded_independent_extent,
            lhs_feat_slice=p.lhs_feat_slice,
            rhs_feat_slice=p.rhs_feat_slice,
            padded_lhs_features=p.padded_lhs_features,
            padded_rhs_features=p.padded_rhs_features,
            bank_bytes=bank_bytes,
            rhs_overhead_bytes=stack_bytes,
        )

    def _describe_matmul_rhs(self, consumer, config, tensor_name, port, buf_dims=None):
        p = config.parameters
        tile_k = int(p.tiling.tile_k)
        tile_n = int(p.tiling.tile_n)
        k_slice = int(p.lhs_feat_slice)
        raw_k = int(p.lhs_slice_raw)
        n_slice = int(p.rhs_feat_slice)
        raw_n = int(p.rhs_slice_raw)
        view = self._io_view(consumer, tensor_name, 'inputs')
        perm = view.get('perm')
        if perm is not None:
            allowed = [list(range(len(perm)))]
            if len(perm) >= 2:
                swap_last_two = list(range(len(perm)))
                swap_last_two[-2], swap_last_two[-1] = swap_last_two[-1], swap_last_two[-2]
                allowed.append(swap_last_two)
            if list(perm) not in allowed:
                raise ValueError(f'{consumer.name}: matmul RHS does not support io_view permutation {perm}.')
        shapes = p.io_shapes['inputs'][tensor_name]
        buffer_order = list(view['buffer_order'])
        buffer_dimension = (
            [int(shapes['padded'][i]) for i in buffer_order] if buf_dims is None else [int(x) for x in buf_dims]
        )
        boundary_dimension = [int(shapes['logical'][i]) for i in buffer_order]
        io_boundary_dimension = list(boundary_dimension)
        feat_dim, indep_dim, traversal_dims = self._canonical_buffer_axes(view, len(buffer_dimension), buffer_order)
        io_tiling_dimension = list(io_boundary_dimension)
        io_tiling_dimension[feat_dim] = raw_n
        io_tiling_dimension[indep_dim] = raw_k
        tiling_dimension = [1 for _ in buffer_dimension]
        tiling_dimension[feat_dim] = tile_n
        tiling_dimension[indep_dim] = tile_k

        row = int(port) // int(p.parallelism.cas_length)
        col = int(port) % int(p.parallelism.cas_length)
        offset = [0 for _ in buffer_dimension]
        offset[feat_dim] = row * n_slice
        offset[indep_dim] = col * k_slice

        tile_traversal = []
        used = {indep_dim, feat_dim}
        tile_traversal.append({'dimension': feat_dim, 'stride': tile_n, 'wrap': max(1, n_slice // tile_n)})
        tile_traversal.append({'dimension': indep_dim, 'stride': tile_k, 'wrap': max(1, k_slice // tile_k)})
        for dim in traversal_dims:
            if dim in used:
                continue
            tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})

        return {
            'access': 'read',
            'buffer_dimension': buffer_dimension,
            'tiling_dimension': tiling_dimension,
            'offset': offset,
            'tile_traversal': tile_traversal,
            'boundary_dimension': boundary_dimension,
            'io_tiling_dimension': io_tiling_dimension,
            'io_boundary_dimension': io_boundary_dimension,
            'slice_dimension': feat_dim,
            'feature_dimension': feat_dim,
            'independent_dimension': indep_dim,
            'packing': 'mmul_rhs',
            'packing_tile_k': tile_k,
            'packing_tile_n': tile_n,
        }

    def footprint(self, context: OpImplPlacementContext) -> OpImplFootprint:
        p = context.config.parameters
        return OpImplFootprint(width=p.parallelism.cas_length, height=p.parallelism.cas_num)
