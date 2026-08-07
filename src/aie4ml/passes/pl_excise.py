# Copyright 2026 Advent Lab, aie4ml

"""Excise layers marked ``run_on='pl'`` from the AIE graph so they can run on the PL."""

from __future__ import annotations

from typing import List

from ..ir import PLCut, get_backend_context
from .base import AIEPass

# Ops that reduce over the feature axis and therefore need COMPLETE feature rows on the PL kernel
# (each PLIO stream must carry whole rows, not feature-halves). Drives the mem-tile row-join in the
# transport materialize pass. Elementwise ops (activations, add) are fine with the default split.
_FEATURE_REDUCTION_OPS = frozenset({'softmax', 'layer_norm'})


class ExcisePLNodes(AIEPass):
    """Cut layers marked ``run_on='pl'`` out of the AIE graph.

    Runs BEFORE ``resolve``, which is what mints execution instances -- so an excised node never
    gets one, and every later stage therefore skips it for free (``if not inst: continue`` in pack,
    placement, transport and the writer's layer collection). The AIE graph is left split in two,
    rejoined through the fabric:

        AIE(... -> dense_1) -> PLIO_ofm -> [PL kernel] -> PLIO_ifm -> AIE(dense_2 -> ...)

    The cut itself is pure boundary bookkeeping: the excised node's INPUT tensor becomes a graph
    OUTPUT and its OUTPUT tensor becomes a graph INPUT. 
    """

    def __init__(self):
        self.name = 'excise_pl_nodes'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical

        pl_nodes = [n for n in graph.nodes if str(n.directives.get('run_on', 'aie')).lower() == 'pl']
        if not pl_nodes:
            return False

        self._check_config(ctx, pl_nodes)
        self._check_not_consecutive(pl_nodes)
        for node in pl_nodes:
            self._excise(graph, node)
        return True

    # -- guards ---------------------------------------------------------------------------

    def _check_config(self, ctx, pl_nodes: List) -> None:
        """Config-level preconditions for offloading anything to the PL. Checked once."""
        names = ', '.join(self._layer(n) for n in pl_nodes)

        target = str(ctx.aie_config.get('Target', 'aie')).lower()
        if target != 'hardware':
            raise ValueError(
                f"run_on='pl' ({names}) requires AIEConfig.Target='hardware', got {target!r}. "
                "With target='aie' no PL is emitted at all, so cutting the graph would leave the "
                'AIE array with PLIO ports that nothing drives.'
            )

        mode = str(ctx.aie_config.get('PLDataMoverMode', 'benchmark')).lower()
        if mode != 'memory_stream':
            raise NotImplementedError(
                f"run_on='pl' ({names}) currently requires PLDataMoverMode='memory_stream', got "
                f"{mode!r}. 'benchmark' uses one combined CU that owns every PLIO port and preloads "
                "all iterations on-chip; 'external_stream' is structurally compatible but not yet "
                'wired up.'
            )

    def _check_not_consecutive(self, pl_nodes: List) -> None:
        """Reject back-to-back PL layers BEFORE mutating anything.
        """
        pl_names = {n.name for n in pl_nodes}
        for node in pl_nodes:
            for tensor in node.inputs:
                producer = tensor.producer
                if producer is not None and producer.name in pl_names:
                    raise NotImplementedError(
                        f'{self._layer(producer)} -> {self._layer(node)}: consecutive PL layers are '
                        'not supported. Each cut must land between two AIE stages.'
                    )

    # -- the cut --------------------------------------------------------------------------

    def _excise(self, graph, node) -> None:
        layer = self._layer(node)

        # Weighted layers (layer_norm gamma/beta, ...) CAN run on the PL: the PL kernel gets its
        # weights baked into the hls4ml firmware (the generated top), and _cut_node drops the param
        # tensors from the AIE graph. Only the ACTIVATION inputs become cut-outs.
        activations = [t for t in node.inputs if not t.is_parameter]
        if not activations or len(node.outputs) != 1:
            raise ValueError(
                f"{layer}: run_on='pl' requires >=1 activation input(s) and a single output; got "
                f'{len(activations)} input(s) and {len(node.outputs)} output(s).'
            )

        out_tv = node.outputs[0]

        # Each input becomes a cut-out (AIE -> PL): validate it is an INTERMEDIATE tensor (an AIE
        # stage in front) with no fanout (not broadcast to AIE and PL at once).
        for in_tv in activations:
            producer = in_tv.producer
            if producer is None or producer.is_placeholder:
                raise ValueError(
                    f"{layer}: run_on='pl' requires an INTERMEDIATE layer, but its input {in_tv.name!r} "
                    'is a graph input -- there is no AIE stage in front of it to cut against. To source '
                    "the array from the fabric instead, use PLDataMoverMode='external_stream'."
                )
            # Fanout at the cut: collect._group_edges keys on consumer_group and a graph-output leg
            # carries the 'graph_output' sentinel, so a shared feed is not yet validated.
            if len(in_tv.consumers) != 1:
                others = [c.name for c in in_tv.consumers if c is not node]
                raise NotImplementedError(
                    f'{layer}: its input {in_tv.name!r} also feeds {others}. Broadcasting a cut tensor '
                    'to the PL and to the AIE at the same time is not yet validated.'
                )

        if out_tv.name in graph.output_tensor_names:
            raise ValueError(
                f"{layer}: run_on='pl' requires an INTERMEDIATE layer, but its output "
                f'{out_tv.name!r} is the model output -- no AIE stage would consume the PL result.'
            )
        if not out_tv.consumers:
            raise ValueError(f'{layer}: output {out_tv.name!r} has no consumers; nothing to cut back into.')

        width = self._cut_width(layer, activations + [out_tv])

        # remove_node(mode='cut') leaves the IR intentionally invalid until the marks land: out_tv is
        # now producer-less, which verify() rejects unless it is a declared graph input.
        graph.remove_node(node, mode='cut')
        for in_tv in activations:
            graph.mark_graph_output(in_tv.name)  # AIE -> PL (one per input)
        graph.mark_graph_input(out_tv.name)      # PL  -> AIE
        graph.pl_cuts.append(
            PLCut(
                node_name=node.name,
                source_layer=layer,
                cut_out_tensors=tuple(t.name for t in activations),
                cut_in_tensor=out_tv.name,
                width=width,
                reduces_features=node.op_type in _FEATURE_REDUCTION_OPS,
            )
        )

        ins = ', '.join(t.name for t in activations)
        print(
            f"[aie4ml] run_on='pl': excised {layer!r} from the AIE graph "
            f'(AIE -> [{ins}] -> PL -> {out_tv.name} -> AIE).'
        )

    @staticmethod
    def _cut_width(layer: str, tensors) -> int:
        """Element width in bits shared by every cut tensor.
        But all widths must agree."""
        widths = []
        for tensor in tensors:
            precision = getattr(tensor, 'precision', None)
            width = getattr(precision, 'width', None)
            if width is None:
                raise ValueError(f'{layer}: tensor {tensor.name!r} has no resolved precision width.')
            widths.append(int(width))
        if len(set(widths)) != 1:
            raise NotImplementedError(
                f'{layer}: a PL cut carries raw AXI-stream bytes, so all cut tensors must share an '
                f'element width; got {widths} for {[t.name for t in tensors]}.'
            )
        return widths[0]

    @staticmethod
    def _layer(node) -> str:
        """User-facing layer name ('softmax_0'), falling back to the IR node name."""
        return str(node.metadata.get('source_layer') or node.name)
