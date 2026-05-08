from __future__ import annotations

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, input_role, input_tensor_for_role
from ...base import OpImplFootprint
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from .common import (
    describe_family_lhs_staging,
    describe_family_rhs_staging,
    validate_family_tile_contract,
)
from .config import MatmulConfig
from .dense import _BaseDenseMatmulVariant

_MATMUL_STACK_OVERHEAD_BYTES = 1024  # kernel stack reserved for matmul RHS packing


@register_variant
class MatmulOpImplVariant(_BaseDenseMatmulVariant):
    variant_id = 'matmul.v1'
    op_type = 'matmul'
    graph_header = 'matmul_graph.h'
    graph_name = 'matmul_graph'
    param_template = 'matmul'
    supported_generations = ('AIE-ML', 'AIE-MLV2')
    supported_precisions = (
        {'lhs': 'int8', 'rhs': 'int8', 'output': 'int8', 'acc': 'int32'},
        {'lhs': 'int8', 'rhs': 'int8', 'output': 'int16', 'acc': 'int32'},
        {'lhs': 'int8', 'rhs': 'int8', 'output': 'int32', 'acc': 'int32'},
        {'lhs': 'int16', 'rhs': 'int8', 'output': 'int8', 'acc': 'int32'},
        {'lhs': 'int16', 'rhs': 'int16', 'output': 'int16', 'acc': 'int64'},
        {'lhs': 'int16', 'rhs': 'int16', 'output': 'int32', 'acc': 'int64'},
        {'lhs': 'bfloat16', 'rhs': 'bfloat16', 'output': 'bfloat16', 'acc': 'accfloat'},
        {'lhs': 'float32', 'rhs': 'float32', 'output': 'float32', 'acc': 'accfloat'},
        {'lhs': 'fp8_e4m3', 'rhs': 'fp8_e4m3', 'output': 'fp8_e4m3', 'acc': 'accfloat'},
    )
    supported_input_modes = ('direct', 'memtile', 'plio', 'auto')
    supported_output_modes = ('direct', 'memtile', 'plio', 'auto')

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, producer=None):
        view = config.io_views[tensor_name]
        role = input_role(node, tensor_name)
        if role == 'rhs':
            return describe_family_rhs_staging(view, config.microtiling, config.parallelism, port, buf_dims)
        return describe_family_lhs_staging(view, config.microtiling, port, buf_dims)

    def pack(self, inst: OpImplInstance):
        return {}

    def get_artifacts(self, inst: OpImplInstance):
        return []

    def validate_config(self, node, config: MatmulConfig, device) -> None:
        p = config
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_view = p.io_views[lhs_tensor.name]
        rhs_view = p.io_views[rhs_tensor.name]
        output_view = p.io_views[node.outputs[0].name]

        rhs_perm = rhs_view.perm
        if rhs_perm is not None:
            rank = len(rhs_perm)
            identity = list(range(rank))
            if rank >= 2:
                swapped = list(range(rank))
                swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
                allowed = [identity, swapped]
            else:
                allowed = [identity]
            if list(rhs_perm) not in allowed:
                raise ValueError(f'{node.name}: matmul RHS does not support io_view permutation {rhs_perm}.')

        rhs_prec = p.precision['rhs']
        rhs_is_float = isinstance(rhs_tensor.precision, FloatIntent)
        if not rhs_is_float and not bool(rhs_prec.signed):
            raise ValueError(f'{node.name}: matmul RHS must use a signed integer precision.')

        rhs_shape = tuple(int(x) for x in rhs_view.logical)
        if len(rhs_shape) < 2:
            raise ValueError(f'{node.name}: matmul RHS must be rank >=2, got {len(rhs_shape)}.')
        validate_family_tile_contract(
            node_name=node.name,
            precision=p.precision,
            parallelism=p.parallelism,
            microtiling=p.microtiling,
            lhs_view=lhs_view,
            output_view=output_view,
            bank_bytes=int(device.bank_mem_bytes),
            rhs_overhead_bytes=_MATMUL_STACK_OVERHEAD_BYTES,
        )

    def footprint(self, node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=config.parallelism.cas_length,
            height=config.parallelism.cas_num,
            extras={'keepout_left': 1},
        )

    def build_ports(self, node, config: MatmulConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding(group='inA', count=int(config.parallelism.cas_length)),
                rhs_tensor.name: PortBinding(
                    group='inB',
                    count=int(config.parallelism.cas_length) * int(config.parallelism.cas_num),
                ),
            },
            outputs={node.outputs[0].name: PortBinding(group='outC', count=int(config.parallelism.cas_num))},
        )
