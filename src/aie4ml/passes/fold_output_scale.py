from __future__ import annotations

from ..ir import TraitInstance, get_backend_context
from .base import AIEPass


class FoldOutputScale(AIEPass):
    """Fold a constant output scale into a Dense/MatMul execution contract."""

    def __init__(self):
        self.name = 'fold_output_scale'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for scale_node in list(graph.nodes):
            if scale_node.op_type != 'scale':
                continue
            if len(scale_node.inputs) != 1 or len(scale_node.outputs) != 1:
                raise ValueError(f'{scale_node.name}: scale must have exactly one input and one output.')

            input_tensor = scale_node.inputs[0]
            producer = input_tensor.producer
            if producer is None or producer.op_type not in ('dense', 'matmul'):
                raise NotImplementedError(
                    f'{scale_node.name}: constant output scale can currently fuse only into Dense or MatMul.'
                )
            if len(input_tensor.consumers) != 1:
                raise NotImplementedError(
                    f'{scale_node.name}: cannot fuse output scale because {input_tensor.name!r} has '
                    f'{len(input_tensor.consumers)} consumers.'
                )

            scale = float(scale_node.metadata['scale'])
            if scale <= 0.0:
                raise ValueError(f'{scale_node.name}: output scale must be positive, got {scale}.')
            existing = producer.traits.get('output_scale')
            combined_scale = scale * float(existing.data['scale']) if existing is not None else scale
            producer.add_trait(TraitInstance('output_scale', {'scale': combined_scale}))
            graph.remove_node(scale_node, mode='contract')
            changed = True

        return changed
