from __future__ import annotations

from typing import Any, Dict, Sequence


def normalized_direct_staging(desc: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if desc is None:
        return None
    data = {k: v for k, v in desc.items() if k not in ('access', 'boundary_dimension')}
    if 'io_boundary_dimension' in data and 'boundary_dimension' not in data:
        data['boundary_dimension'] = data['io_boundary_dimension']
    return data


def direct_transport_supported(
    ctx, producer, consumer, tensor_name: str, producer_ports: Sequence[int], consumer_ports: Sequence[int]
) -> bool:
    if producer is None or consumer is None or len(producer_ports) != len(consumer_ports):
        return False

    src_inst = ctx.ir.execution.get(producer.name)
    dst_inst = ctx.ir.execution.get(consumer.name)
    if src_inst is None or dst_inst is None:
        raise RuntimeError(f'{tensor_name}: direct transport legality requires resolved execution instances.')

    for p_port, c_port in zip(producer_ports, consumer_ports):
        src_desc = src_inst.variant.describe_output_staging(producer, src_inst.config, tensor_name, int(p_port), None)
        dst_desc = dst_inst.variant.describe_input_staging(
            consumer,
            dst_inst.config,
            tensor_name,
            int(c_port),
            None,
            producer,
        )
        if normalized_direct_staging(src_desc) != normalized_direct_staging(dst_desc):
            return False
    return True
