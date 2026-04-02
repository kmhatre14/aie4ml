from ..ir import TraitInstance, get_backend_context
from .base import AIEPass


class FoldTransposeViews(AIEPass):
    def __init__(self):
        self.name = 'fold_transpose_views'

    def _io_view_data(self, node):
        trait = node.traits.get('io_view')
        if trait is None:
            trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
            node.add_trait(trait)
        trait.data.setdefault('inputs', {})
        trait.data.setdefault('outputs', {})
        return trait.data

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for node in list(graph.nodes):
            if node.op_type != 'transpose':
                continue

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
                'layout': 'channels_last',
                'independent_axes': [int(i) for i in range(rank - 1)],
                'buffer_order': [int(i) for i in reversed(range(rank))],
                'perm': [int(p) for p in perm],
            }

            for consumer in out_tv.consumers:
                cons_view = self._io_view_data(consumer)
                cons_view['inputs'][in_tv.name] = dict(in_view)
                cons_view['inputs'].pop(out_tv.name, None)

            graph.remove_node(node, mode='bypass')
            changed = True

        return changed
