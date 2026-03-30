"""Dense+activation fusion pass on aie4ml logical IR."""

from ..ir import TraitInstance, get_backend_context
from .base import AIEPass


class FuseActivationCasts(AIEPass):
    """Fuse relu into Dense (adds fused_activation trait) and contract linear activations out.

    After this pass, no activation nodes remain in the IR and output tensors carry post-activation precision.
    """

    def __init__(self):
        self.name = 'fuse_activation_casts'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for act_node in list(graph.nodes):
            if act_node.op_type != 'activation' or len(act_node.inputs) != 1 or len(act_node.outputs) != 1:
                continue

            activation = (act_node.metadata.get('activation', '') or '').lower()

            if activation == 'relu':
                in_tensor = act_node.inputs[0]
                producer = in_tensor.producer
                if producer is None or producer.op_type != 'dense' or len(producer.outputs) != 1:
                    continue
                producer.add_trait(TraitInstance('fused_activation', {'activation': 'relu'}))
                graph.remove_node(act_node, mode='contract')
                changed = True

            elif activation in ('linear', ''):
                # Pure precision cast — eliminate regardless of what precedes it.
                graph.remove_node(act_node, mode='contract')
                changed = True

        return changed
