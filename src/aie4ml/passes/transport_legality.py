from __future__ import annotations

import copy
from typing import Sequence

from ..op_impls.utils.io import normalized_staging


def _localize_descriptor(desc, base: Sequence[int], buffer_dimension: Sequence[int]):
    if not base:
        return desc

    localized = copy.deepcopy(desc)
    offset = [int(x) for x in localized['offset']]
    dims = [int(x) for x in buffer_dimension]
    if len(offset) != len(base) or len(offset) != len(dims):
        raise RuntimeError('direct transport descriptor rank mismatch during leg localization.')

    localized['offset'] = [offset[dim] - int(base[dim]) for dim in range(len(offset))]
    localized['buffer_dimension'] = list(dims)
    for key in ('boundary_dimension', 'io_boundary_dimension'):
        if key not in localized:
            continue
        boundary = [int(x) for x in localized[key]]
        if len(boundary) != len(dims):
            raise RuntimeError('direct transport boundary rank mismatch during leg localization.')
        localized[key] = [min(dims[dim], max(0, boundary[dim] - int(base[dim]))) for dim in range(len(dims))]
    return localized


def direct_transport_supported(
    ctx,
    producer,
    consumer,
    logical_tensor: str,
    producer_tensor: str,
    consumer_tensor: str,
    producer_ports: Sequence[int],
    consumer_ports: Sequence[int],
    consumer_offset_base: Sequence[int] = (),
) -> bool:
    if producer is None or consumer is None or len(producer_ports) != len(consumer_ports):
        return False

    src_inst = ctx.ir.execution.get(producer.name)
    dst_inst = ctx.ir.execution.get(consumer.name)
    if src_inst is None or dst_inst is None:
        raise RuntimeError(f'{logical_tensor}: direct transport legality requires resolved execution instances.')

    tc = ctx.ir.execution.tensor_contracts.get(producer_tensor)
    if tc is not None:
        if len(producer_ports) != len(tc.port_staging) or len(consumer_ports) != len(tc.port_staging):
            return False
        if src_inst.io_views.get(producer_tensor) is None or dst_inst.io_views.get(consumer_tensor) is None:
            return False

    for p_port, c_port in zip(producer_ports, consumer_ports):
        src_desc = src_inst.variant.describe_output_staging(
            producer, src_inst.config, producer_tensor, int(p_port), None
        )
        dst_desc = dst_inst.variant.describe_input_staging(
            consumer,
            dst_inst.config,
            consumer_tensor,
            int(c_port),
            None,
            producer,
        )
        dst_desc = _localize_descriptor(dst_desc, consumer_offset_base, src_desc['buffer_dimension'])
        if normalized_staging(src_desc) != normalized_staging(dst_desc):
            return False
    return True
