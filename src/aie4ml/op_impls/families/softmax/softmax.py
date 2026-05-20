from __future__ import annotations

from typing import Any, Dict

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ...base import OpImplFootprint, OpImplVariant
from ...registry import register_variant
from ..elementwise.common import describe_elementwise_staging
from .common import pack_hccs_params, validate_softmax_tile_contract
from .config import SoftmaxConfig


@register_variant
class SoftmaxHccsI8OpImplVariant(OpImplVariant):
    """HCCS integer Softmax surrogate.

    Implements Head-Calibrated Clipped-Linear Softmax, an integer attention-softmax
    surrogate using calibrated B/S/Dmax parameters instead of exponentials.
    This variant is intended for QAT/calibrated models and is not a drop-in
    replacement for generic floating-point ONNX Softmax. See https://arxiv.org/pdf/2604.02292v1
    """

    variant_id = 'softmax.hccs.i8.v1'
    op_type = 'softmax'
    graph_header = 'softmax_graph.h'
    graph_name = 'softmax_hccs_graph'
    param_template = 'softmax'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = (
        {'lhs': 'int8', 'output': 'int8', 'B': 'int16', 'S': 'int8', 'Dmax': 'int8'},
        {'lhs': 'int8', 'output': 'int16', 'B': 'int16', 'S': 'int8', 'Dmax': 'int8'},
    )
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def build_template_params(self, node: OpNode, config: SoftmaxConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        in_view = config.io_views[in_tensor.name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            rows=int(in_view.compacted_tile_outer),
            cols=int(in_view.full_inner),
        )
        params['packed_hccs'] = pack_hccs_params(
            config.hccs,
            param_sets=int(config.param_sets),
            cols=int(in_view.full_inner),
            cas_num=int(config.parallelism.cas_num),
        )
        return params

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'read', 'outer', buf_dims)

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        return describe_elementwise_staging(config.io_views[tensor_name], port, 'write', 'outer', buf_dims)

    def output_staging_contract(self, node, config: SoftmaxConfig, tensor_name: str):
        return 'outer'

    def validate_config(self, node: OpNode, config: SoftmaxConfig, device) -> None:
        lhs_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        validate_softmax_tile_contract(
            node_name=node.name,
            precision=config.precision,
            rows=int(lhs_view.compacted_tile_outer),
            cols=int(lhs_view.full_inner),
            bank_bytes=int(device.bank_mem_bytes),
            vec_size=int(config.vec_size),
        )
        if int(config.inv_shift) < 1 or int(config.inv_shift) > 30:
            raise ValueError(f'{node.name}: HCCS Softmax inv_shift must be in [1, 30], got {config.inv_shift}.')
        if int(lhs_view.compacted_tile_outer) * int(config.parallelism.cas_num) != int(lhs_view.compacted_full_outer):
            raise ValueError(
                f'{node.name}: compacted_full_outer={lhs_view.compacted_full_outer} must equal '
                f'compacted_tile_outer * cas_num ({lhs_view.compacted_tile_outer} * {config.parallelism.cas_num}).'
            )
        pack_hccs_params(
            config.hccs,
            param_sets=int(config.param_sets),
            cols=int(lhs_view.full_inner),
            cas_num=int(config.parallelism.cas_num),
        )

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def footprint(self, node: OpNode, config: SoftmaxConfig) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=int(config.parallelism.cas_num), extras={'keepout_left': 1})

    def build_ports(self, node: OpNode, config: SoftmaxConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        return super().build_ports(
            node,
            {in_tensor.name: int(config.parallelism.cas_num)},
            int(config.parallelism.cas_num),
        )
