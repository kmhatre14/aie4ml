from __future__ import annotations

import copy

from ..ir import get_backend_context
from .base import AIEPass


class LegalizeFanoutEntries(AIEPass):
    def __init__(self):
        self.name = 'legalize_fanout_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        entries = state['entries']

        changed = False
        rewritten = []

        for entry in entries:
            needs_consumer_split = len(entry.consumers) > 1
            needs_output_split = entry.graph_output and bool(entry.consumers)

            if not needs_consumer_split and not needs_output_split:
                rewritten.append(entry)
                continue

            changed = True

            if needs_output_split:
                output_only = copy.copy(entry)
                output_only.consumers = []
                output_only.graph_output = True
                rewritten.append(output_only)

            for conn in entry.consumers:
                replica = copy.copy(entry)
                replica.consumers = [conn]
                replica.graph_output = False
                rewritten.append(replica)

        state['entries'] = rewritten

        for entry in state['entries']:
            if len(entry.consumers) > 1:
                raise ValueError(f'{entry.tensor}: fanout legalization incomplete.')
            if entry.consumers and entry.graph_output:
                raise ValueError(f'{entry.tensor}: output leg was not separated from consumer leg.')

        return changed
