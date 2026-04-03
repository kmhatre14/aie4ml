from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import (
    OpImplConfig,
    OpImplFootprint,
    OpImplPlacementContext,
    OpImplSelectionContext,
    OpImplVariant,
)
from .common import (
    TILING_OPTIONS,
    describe_family_lhs_staging,
    describe_family_output_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    select_generation_key,
    tiling_key,
    validate_family_tile_contract,
)
from .types import MatmulParallelismConfig, MatmulTilingConfig


@dataclass(frozen=True)
class DenseFlags:
    """Dense implementation behaviour flags used by templates and codegen."""

    use_relu: bool
    transpose_input: bool
    use_bias: bool


@dataclass(frozen=True)
class DenseOpImplParameters:
    """Dense-specific compile-time parameters for the selected AIE implementation."""

    precision: Dict[str, Any]
    parallelism: MatmulParallelismConfig
    tiling: MatmulTilingConfig
    flags: DenseFlags
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


class DenseOpImplVariant(OpImplVariant):
    def _io_view(self, node: OpNode, tensor_name: str, direction: str) -> Dict[str, Any]:
        return node.traits['io_view'].data[direction][tensor_name]

    def _map_view_axis(self, view: Dict[str, Any], axis: int) -> int:
        perm = view.get('perm')
        if perm is None:
            return int(axis)
        return perm[axis]

    def _canonical_buffer_axes(
        self, view: Dict[str, Any], rank: int, buffer_order: List[int]
    ) -> Tuple[int, int, List[int]]:
        feat_axis = self._map_view_axis(view, rank - 1)
        indep_axis = self._map_view_axis(view, rank - 2)
        feat_dim = buffer_order.index(int(feat_axis))
        indep_dim = buffer_order.index(int(indep_axis))
        tail_dims = sorted(dim for dim in range(rank) if dim not in (feat_dim, indep_dim))
        return feat_dim, indep_dim, [feat_dim, indep_dim] + tail_dims

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        return self._describe_dense_ifm(node, config, tensor_name, port, buf_dims)

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        return self._describe_dense_ofm(node, config, tensor_name, port, buf_dims)

    def supports(self, context: OpImplSelectionContext) -> bool:
        if not super().supports(context):
            return False

        attrs = context.attributes
        tile_m = int(attrs.tiling['tile_m'])
        tile_k = int(attrs.tiling['tile_k'])
        tile_n = int(attrs.tiling['tile_n'])
        lhs_key = tiling_key(attrs.numeric['lhs'])
        rhs_key = tiling_key(attrs.numeric['rhs'])

        if not all((tile_m, tile_k, tile_n, lhs_key, rhs_key)):
            return False
        return (tile_m, tile_k, tile_n) in self.tiling_options(context.device_generation, (lhs_key, rhs_key))

    def build_config(self, context: OpImplSelectionContext) -> OpImplConfig:
        attrs = context.attributes
        parallel = dict(attrs.parallelism)
        tiling = dict(attrs.tiling)
        slices = dict(attrs.slices)
        scalars = dict(attrs.scalars)
        flags = dict(attrs.flags)
        lhs_tensor = input_tensor_for_role(context.node, 'lhs')
        input_view = self._io_view(context.node, lhs_tensor.name, 'inputs')
        perm = input_view.get('perm')
        transpose_input = perm is not None and perm[-1] != (len(perm) - 1)
        cas = int(parallel['cas_length'])
        chains = int(parallel['cas_num'])

        params = DenseOpImplParameters(
            precision={
                key: attrs.numeric.get(key)
                for key in ('lhs', 'rhs', 'output', 'bias', 'acc')
                if attrs.numeric.get(key) is not None
            },
            parallelism=MatmulParallelismConfig(
                cas_length=cas,
                cas_num=chains,
            ),
            tiling=MatmulTilingConfig(
                tile_m=int(tiling['tile_m']),
                tile_k=int(tiling['tile_k']),
                tile_n=int(tiling['tile_n']),
            ),
            flags=DenseFlags(
                use_relu=bool(flags['use_relu']),
                transpose_input=bool(transpose_input),
                use_bias=bool(context.node.metadata['use_bias']),
            ),
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

        return OpImplConfig(
            variant_id=self.variant_id,
            param_template='dense_bias_relu',
            graph_header='dense_bias_relu_graph.h',
            graph_name='dense_bias_relu_graph',
            parameters=params,
            ports=self._build_port_map(context, cas, chains),
            io_route=dict(attrs.io_route),
        )

    def tiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(TILING_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        from ....aie_types import FloatFormat, FloatIntent
        from .common import pack_as_float as _pack_as_float
        from .common import pack_mmul_rhs_matrix, pack_vector_by_n_slice
        from .common import quantize_to_int as _quantize_to_int

        p = inst.config.parameters
        input_tensor = inst.node.inputs[0]
        weight_tensor = inst.node.inputs[1]
        bias_tensor = inst.node.inputs[2] if len(inst.node.inputs) > 2 else None

        wi = weight_tensor.precision
        if isinstance(wi, FloatIntent):
            W = _pack_as_float(weight_tensor.data, wi.format)
            b = np.asarray(bias_tensor.data, dtype=np.float32) if bias_tensor is not None else None
        else:
            W = _quantize_to_int(
                weight_tensor.data,
                wi.frac,
                wi.width,
                signed=wi.signed,
                rounding_mode=wi.rounding,
                saturation_mode=wi.saturation,
            )
            if bias_tensor is not None:
                bi = bias_tensor.precision
                accum_frac = input_tensor.precision.frac + wi.frac
                b = _quantize_to_int(
                    bias_tensor.data,
                    accum_frac,
                    32,
                    signed=bi.signed,
                    rounding_mode=bi.rounding,
                    saturation_mode=bi.saturation,
                )
            else:
                b = None

        W = np.asarray(W)
        if W.ndim < 2:
            raise ValueError(f'{inst.name}: weight matrix must have at least 2 dimensions, got {W.ndim}.')
        n_in = int(W.shape[-2])
        n_out = int(W.shape[-1])

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=p.lhs_feat_slice,
            N_slice=p.rhs_feat_slice,
            tile_k=p.tiling.tile_k,
            tile_n=p.tiling.tile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=p.parallelism.cas_num,
            dtype=np_dtype_for_spec(p.precision['rhs']),
        )
        if isinstance(wi, FloatIntent) and wi.format == FloatFormat.BF16:
            packed_W = (packed_W.astype(np.uint32) << 16).view(np.float32)
        packed_B = (
            pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=p.rhs_feat_slice,
                cas_num=p.parallelism.cas_num,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            if b is not None
            else None
        )
        return {'packed_weights': packed_W, 'packed_bias': packed_B}

    def _describe_dense_ofm(self, node, config, tensor_name, port, buf_dims=None):
        return describe_family_output_staging(self, node, config, tensor_name, port, buf_dims)

    def _describe_dense_ifm(self, consumer, config, tensor_name, port, buf_dims=None):
        return describe_family_lhs_staging(self, consumer, config, tensor_name, port, buf_dims)

    def validate_config(self, context: OpImplSelectionContext, config: OpImplConfig) -> None:
        p = config.parameters
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
        )

    def footprint(self, context: OpImplPlacementContext) -> OpImplFootprint:
        p = context.config.parameters
        return OpImplFootprint(
            width=p.parallelism.cas_length,
            height=p.parallelism.cas_num,
            extras={'keepout_left': 1},
        )

    def get_artifacts(self, inst: OpImplInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config.parameters
        artifacts = [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': p.precision['rhs'].c_type,
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            }
        ]
        packed_bias = inst.artifacts.get('packed_bias')
        if packed_bias is None:
            # The current dense graph always exposes a bias RTP port. For biasless
            # layers we feed an explicit zero bias matching the fixed kernel
            # interface instead of trying to specialize the graph signature.
            packed_bias = np.zeros(
                (int(p.parallelism.cas_num), int(p.rhs_feat_slice)),
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
        artifacts.append(
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': packed_bias,
                'dtype': p.precision['bias'].c_type,
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            }
        )
        return artifacts
