from __future__ import annotations

from ..ir import TraitInstance, get_backend_context
from .base import AIEPass


class FoldViewOps(AIEPass):
    """Lower semantic view ops into explicit tensor view traits."""

    def __init__(self):
        self.name = 'fold_view_ops'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for node in list(graph.nodes):
            if node.op_type == 'transpose':
                changed = self._fold_transpose(graph, node) or changed
            elif node.op_type == 'concat':
                changed = self._fold_concat(node) or changed

        return changed

    def _io_view_data(self, node):
        trait = node.traits.get('io_view')
        if trait is None:
            trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
            node.add_trait(trait)
        trait.data.setdefault('inputs', {})
        trait.data.setdefault('outputs', {})
        return trait.data

    def _fold_transpose(self, graph, node) -> bool:
        in_tv = node.inputs[0]
        out_tv = node.outputs[0]

        perm = node.metadata.get('perm')
        if perm is None:
            raise ValueError(f'{node.name}: missing transpose permutation metadata.')
        perm = [int(x) for x in perm]

        rank = len(in_tv.shape)
        if len(out_tv.shape) != rank:
            raise ValueError(f'{node.name}: transpose rank mismatch between input and output.')
        if sorted(perm) != list(range(rank)):
            raise ValueError(f'{node.name}: invalid permutation {perm} for rank {rank}.')
        if (node.metadata.get('data_format', 'channels_last') or '').lower() != 'channels_last':
            raise ValueError(f'{node.name}: only channels_last transpose is supported.')

        in_view = {
            'buffer_order': [int(i) for i in reversed(range(rank))],
            'perm': [int(p) for p in perm],
        }

        for consumer in out_tv.consumers:
            cons_view = self._io_view_data(consumer)
            cons_view['inputs'][in_tv.name] = dict(in_view)
            cons_view['inputs'].pop(out_tv.name, None)
            if out_tv.name in consumer.roles:
                consumer.roles[in_tv.name] = consumer.roles.pop(out_tv.name)

        graph.remove_node(node, mode='bypass')
        return True

    def _fold_concat(self, node) -> bool:
        if len(node.outputs) != 1:
            raise ValueError(f'{node.name}: concat must have exactly one output.')
        if not node.inputs:
            raise ValueError(f'{node.name}: concat must have at least one input.')

        output = node.outputs[0]
        axis = int(node.metadata.get('axis', -1))
        rank = len(output.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(f'{node.name}: concat axis {node.metadata.get("axis")} is out of range for rank {rank}.')

        offset = 0
        slices = []
        for tensor in node.inputs:
            if len(tensor.shape) != rank:
                raise ValueError(f'{node.name}: concat input {tensor.name!r} rank does not match output rank.')
            extent = int(tensor.shape[axis])
            slices.append({'input': tensor.name, 'start': int(offset), 'extent': extent})
            offset += extent

        if int(output.shape[axis]) != int(offset):
            raise ValueError(
                f'{node.name}: concat output axis extent {output.shape[axis]} does not match input sum {offset}.'
            )

        node.add_trait(
            TraitInstance(
                'concat_view',
                {
                    'kind': 'concat',
                    'axis': axis,
                    'output': output.name,
                    'slices': slices,
                },
            )
        )
        node.is_placeholder = True
        return True
