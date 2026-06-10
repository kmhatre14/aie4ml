from __future__ import annotations

import copy
from typing import Any, Dict, MutableMapping, Sequence


def rebase_descriptor_offset(descriptor: MutableMapping[str, Any], offset_base: Sequence[int]) -> None:
    """Rebase only a descriptor offset into endpoint-local coordinates."""
    if not offset_base:
        return
    offset = [int(value) for value in descriptor['offset']]
    base = [int(value) for value in offset_base]
    if len(offset) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during offset rebasing ({len(offset)} != {len(base)}).')
    descriptor['offset'] = [offset[dim] - base[dim] for dim in range(len(offset))]


def localize_descriptor(
    descriptor: MutableMapping[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> None:
    """Rebase a descriptor and its boundaries into an endpoint-local buffer."""
    if not offset_base:
        return

    dims = [int(value) for value in buffer_dimension]
    rebase_descriptor_offset(descriptor, offset_base)
    base = [int(value) for value in offset_base]
    if len(dims) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during localization ({len(dims)} != {len(base)}).')

    descriptor['buffer_dimension'] = list(dims)
    for key in ('boundary_dimension', 'io_boundary_dimension'):
        if key not in descriptor:
            continue
        boundary = [int(value) for value in descriptor[key]]
        if len(boundary) != len(dims):
            raise RuntimeError(f'descriptor {key} rank mismatch during localization.')
        descriptor[key] = [min(dims[dim], max(0, boundary[dim] - base[dim])) for dim in range(len(dims))]


def localized_graph_io_descriptor(
    descriptor: Dict[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> Dict[str, Any]:
    """Return a graph-IO descriptor localized to one memory-tile shard."""
    localized = copy.deepcopy(descriptor)
    dims = [int(value) for value in buffer_dimension]
    boundary = list(localized['io_boundary_dimension'])
    base = [int(value) for value in offset_base]
    if len(base) != len(dims) or len(boundary) != len(dims):
        raise RuntimeError('graph-IO descriptor rank mismatch during localization.')
    rebase_descriptor_offset(localized, base)
    localized['buffer_dimension'] = list(dims)
    if localized.get('access') == 'read':
        localized['boundary_dimension'] = [
            min(dims[dim], max(0, int(boundary[dim]) - base[dim])) for dim in range(len(dims))
        ]
    else:
        localized.pop('boundary_dimension', None)
    return localized
