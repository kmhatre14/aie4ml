# Copyright 2026 Advent Lab, aie4ml

"""PL layer offload: partition the PLIO ports, and spec the PL compute kernels.

When a layer carries ``run_on='pl'`` the ExcisePLNodes pass cuts it out of the AIE graph, 
turning its two boundary tensors into extra PLIO ports. The array then has
more PLIO than the model's own inputs and outputs -- and the two roles are INTERLEAVED, not
appended:

    PLIO_ifm:  [0][1][2][3] [4][5]          PLIO_ofm:  [0][1] [2]
               \\___model__/ \\_cut_/                    \\_cut_/ \\model/
                  -> mm2s     <- PL kernel               -> PL     -> s2mm

  * this module  -- port ROLES + each PL compute kernel's spec
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# PL compute kernels live under pl/compute/
_COMPUTE_TEMPLATE_DIR = 'pl/compute'
# The generic AXIS wrapper around an hls4ml-generated op body (pl_hls4ml.py fills the firmware +
# template vars at write time).
_HLS4ML_KERNEL = 'hls4ml_kernel'

# CU names for data movers and tick gen
_RESERVED_CU_NAMES = frozenset({'mm2s', 's2mm', 'tick_gen', 'traffic_gen', 'ddr_pl_aie_datamover'})


@dataclass(frozen=True)
class PLKernelSpec:
    """One PL compute kernel standing in for an excised AIE layer."""

    name: str  # CU name == the layer name, e.g. 'softmax_0'
    source_layer: str
    # One PLIO_ofm port group per OP INPUT (AIE -> PL). A single-input op (softmax) has one group; a
    # multi-input op (add) has one per operand. Each group is paired shard-for-shard with the output
    # group, so group[k] and cut_in_ports[k] describe the same shard of one row.
    cut_out_port_groups: List[List[int]]
    cut_in_ports: List[int]  # kernel.s_out_*   -> PLIO_ifm        (PL -> AIE)
    beats_per_iter: int  # AXIS beats ONE stream carries per iteration
    cpp_template: str
    cfg_template: str

    @property
    def n_op_inputs(self) -> int:
        """Number of distinct operand tensors (1 for softmax, 2 for add)."""
        return len(self.cut_out_port_groups)

    @property
    def cut_out_ports(self) -> List[int]:
        """Flat PLIO_ofm list in s_in_* order: input-0 shards, then input-1 shards, ... Consumed by
        the system.cfg wiring (sc= lines map flat index -> s_in_i)."""
        return [p for group in self.cut_out_port_groups for p in group]

    @property
    def n_in(self) -> int:
        return len(self.cut_out_ports)

    @property
    def n_out(self) -> int:
        return len(self.cut_in_ports)

    @property
    def shards_per_input(self) -> int:
        """PLIO streams carrying one operand (== the output shard count; the wrapper pairs them)."""
        return len(self.cut_in_ports)


@dataclass(frozen=True)
class BoundaryPlan:
    """PLIO ports partitioned into the model boundary (DDR movers) and the PL-kernel cuts."""

    # Model-boundary tensors 
    model_input_tensors: List[str]  # -> one mm2s each
    model_output_tensors: List[str]  # -> one s2mm each
    kernels: List[PLKernelSpec]

    @property
    def has_cuts(self) -> bool:
        return bool(self.kernels)


def resolve_pl_offload(model_or_ctx, layout) -> BoundaryPlan:
    """Partition the PLIO ports and describe each PL compute kernel.

    ``layout`` is a :class:`simulation.IOLayout` built from the physical plan; its per-tensor port
    lists are already sorted by PLIO index.

    Every graph tensor that is not a cut tensor is a MODEL tensor and gets its own DDR mover, so
    N model inputs / M model outputs are supported (memory_stream). With no cuts every layout
    tensor is a model tensor and an AIE-only hardware build behaves exactly as before.
    """
    ctx = get_backend_context(model_or_ctx)
    cuts = list(ctx.ir.logical.pl_cuts)

    cut_in_names = ctx.ir.logical.cut_in_tensor_names()  # PL -> AIE (extra graph INPUTS)
    cut_out_names = ctx.ir.logical.cut_out_tensor_names()  # AIE -> PL (extra graph OUTPUTS)

    # Mover j serves model tensor j: order by first PLIO port so the numbering is deterministic and
    # matches the host's buffer order (pack_host_data) and data.h's ifm_ports/ofm_ports.
    def _first_port(ports):
        return min(int(p.port) for p in ports)

    model_inputs = sorted((t for t in layout.inputs if t not in cut_in_names), key=lambda t: _first_port(layout.inputs[t]))
    model_outputs = sorted((t for t in layout.outputs if t not in cut_out_names), key=lambda t: _first_port(layout.outputs[t]))
    if not model_inputs or not model_outputs:
        raise RuntimeError(
            'system I/O plan needs at least one model input and one model output tensor. After '
            f'setting aside {len(cuts)} PL cut(s), it found input(s)={model_inputs} and '
            f'output(s)={model_outputs}.'
        )

    return BoundaryPlan(
        model_input_tensors=model_inputs,
        model_output_tensors=model_outputs,
        kernels=[_kernel_spec(ctx, layout, cut) for cut in cuts],
    )


def _kernel_spec(ctx, layout, cut) -> PLKernelSpec:
    """Describe the PL kernel that replaces one excised layer (single- or multi-input)."""
    # One PLIO_ofm port group per OP INPUT (AIE -> PL); the single output group (PL -> AIE).
    out_groups = [layout.outputs.get(t) for t in cut.cut_out_tensors]
    in_ports = layout.inputs.get(cut.cut_in_tensor)  # PL -> AIE
    missing = [t for t, g in zip(cut.cut_out_tensors, out_groups) if not g]
    if missing or not in_ports:
        raise RuntimeError(
            f'{cut.source_layer}: cut tensors did not materialize as PLIO ports '
            f'(missing out={missing}, in={cut.cut_in_tensor!r} -> {bool(in_ports)}). '
            'The physical plan and the recorded cut disagree.'
        )

    # every input group must have the same port count as the output. Port counts follow 
    # the producers' output sharding (each operand's CAS_NUM) and the consumer's input 
    # sharding (CAS_LENGTH); reject a mismatch
    n_out = len(in_ports)
    for tname, group in zip(cut.cut_out_tensors, out_groups):
        if len(group) != n_out:
            raise NotImplementedError(
                f'{cut.source_layer}: operand {tname!r} arrives on {len(group)} PLIO port(s) but the '
                f'result re-enters on {n_out}.'
            )

    # All operands + the result share a shape here (elementwise/reduction), so beats/stream agree.
    beats = _beats_per_iter(cut.source_layer, 'PL->AIE', ctx, in_ports)
    for tname, group in zip(cut.cut_out_tensors, out_groups):
        b = _beats_per_iter(cut.source_layer, f'AIE->PL[{tname}]', ctx, group)
        if b != beats:
            # Caught here rather than in hw_emu, where a volume mismatch presents as an unexplained hang.
            raise NotImplementedError(
                f'{cut.source_layer}: operand {tname!r} carries {b} beat(s)/stream but the result '
                f'{beats}. The PL kernel would have to change the data volume.'
            )

    name = sanitize_identifier(cut.source_layer)
    if name in _RESERVED_CU_NAMES:
        raise ValueError(
            f'{cut.source_layer}: a PL layer cannot be named {name!r} -- that CU name is reserved '
            'for a data mover, and system.cfg would declare two kernels under one nk= name.'
        )

    return PLKernelSpec(
        name=name,
        source_layer=cut.source_layer,
        cut_out_port_groups=[[p.port for p in group] for group in out_groups],
        cut_in_ports=[p.port for p in in_ports],
        beats_per_iter=beats,
        cpp_template=f'{_COMPUTE_TEMPLATE_DIR}/{_HLS4ML_KERNEL}.cpp.jinja',
        cfg_template=f'{_COMPUTE_TEMPLATE_DIR}/{_HLS4ML_KERNEL}.cfg.jinja',
    )


def _beats_per_iter(layer: str, direction: str, ctx, ports) -> int:
    """AXIS beats ONE stream carries per iteration.

    The whole tensor is striped round-robin across its PLIO ports, so each stream moves
    total_bytes / n_streams, and each beat moves PLIOWidthBits/8 bytes.

    tutorial_4 cut: 256 batch x 128 feat, int8 = 32768 B over 2 ports at 16 B/beat -> 1024.
    """
    port0 = ports[0]
    elems = int(math.prod(port0.numpy_boundary_shape))  # the FULL tensor, not one port's slice
    elem_bytes = int(port0.dtype.width) // 8
    total_bytes = elems * elem_bytes

    beat_bytes = int(ctx.device.plio_width_bits) // 8
    n_streams = len(ports)
    divisor = n_streams * beat_bytes
    if total_bytes % divisor != 0:
        raise NotImplementedError(
            f'{layer} ({direction}): the cut tensor is {total_bytes} B, which does not divide evenly '
            f'into {n_streams} stream(s) x {beat_bytes} B/beat. The PL kernel would need a partial '
            'beat, which v1 does not emit.'
        )
    return total_bytes // divisor
