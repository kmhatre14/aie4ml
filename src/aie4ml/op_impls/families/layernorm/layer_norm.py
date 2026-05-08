from __future__ import annotations

from typing import Any, Dict

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import OpImplFootprint, OpImplVariant
from ...registry import register_variant
from ..elementwise.common import describe_elementwise_staging
from .common import (
    BETA_FRAC_BITS,
    GAMMA_FRAC_BITS,
    pack_layernorm_param,
    validate_layernorm_tile_contract,
)
from .config import LayerNormConfig


@register_variant
class LayerNormI8OpImplVariant(OpImplVariant):
    variant_id = 'layer_norm.i8.v1'
    op_type = 'layer_norm'
    graph_header = 'layer_norm_graph.h'
    graph_name = 'layer_norm_graph'
    param_template = 'layer_norm'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = ({'lhs': 'int8', 'output': 'int8', 'gamma': 'int16', 'beta': 'int16'},)
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def build_template_params(self, node: OpNode, config: LayerNormConfig):
        return {f: getattr(config, f) for f in config.__dataclass_fields__}

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'read', 'outer', buf_dims)

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'write', 'outer', buf_dims)

    def output_staging_contract(self, node, config: LayerNormConfig, tensor_name: str):
        return 'outer'

    def validate_config(self, node: OpNode, config: LayerNormConfig, device) -> None:
        validate_layernorm_tile_contract(
            node_name=node.name,
            precision=config.precision,
            tile_outer=int(config.rows),
            full_inner=int(config.cols),
            bank_bytes=int(device.bank_mem_bytes),
            vec_size=int(config.vec_size),
        )
        outer_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        if int(outer_view.compacted_full_outer) != int(config.rows) * int(config.parallelism.cas_num):
            raise ValueError(
                f'{node.name}: compacted_full_outer={outer_view.compacted_full_outer} must equal rows * cas_num '
                f'({config.rows} * {config.parallelism.cas_num}).'
            )
        if int(config.precision['gamma'].frac) != GAMMA_FRAC_BITS:
            raise ValueError(
                f'{node.name}: layernorm_i8 requires gamma in Q{GAMMA_FRAC_BITS} '
                f'(frac={GAMMA_FRAC_BITS}), got frac={config.precision["gamma"].frac}.'
            )
        if int(config.precision['beta'].frac) != BETA_FRAC_BITS:
            raise ValueError(
                f'{node.name}: layernorm_i8 requires beta in Q{BETA_FRAC_BITS} '
                f'(frac={BETA_FRAC_BITS}), got frac={config.precision["beta"].frac}.'
            )

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        p = inst.config
        gamma_tensor = input_tensor_for_role(inst.node, 'gamma')
        beta_tensor = input_tensor_for_role(inst.node, 'beta')
        cas_num = int(p.parallelism.cas_num)
        return {
            'packed_gamma': pack_layernorm_param(
                gamma_tensor.data,
                name='gamma',
                full_inner=int(p.cols),
                frac=int(p.precision['gamma'].frac),
                cas_num=cas_num,
            ),
            'packed_beta': pack_layernorm_param(
                beta_tensor.data,
                name='beta',
                full_inner=int(p.cols),
                frac=int(p.precision['beta'].frac),
                cas_num=cas_num,
            ),
        }

    def get_artifacts(self, inst: OpImplInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        return [
            {
                'name': 'gamma',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_gamma'],
                'dtype': p.precision['gamma'].c_type,
                'storage_dtype': p.precision['gamma'].storage_dtype,
                'filename': f'gamma_{inst_name}.h',
                'port': 'gamma',
            },
            {
                'name': 'beta',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_beta'],
                'dtype': p.precision['beta'].c_type,
                'storage_dtype': p.precision['beta'].storage_dtype,
                'filename': f'beta_{inst_name}.h',
                'port': 'beta',
            },
        ]

    def footprint(self, node: OpNode, config: LayerNormConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=int(config.parallelism.cas_num), extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: LayerNormConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        return super().build_ports(
            node,
            {in_tensor.name: int(config.parallelism.cas_num)},
            int(config.parallelism.cas_num),
        )
