# Copyright 2026 Advent Lab, aie4ml

"""PL layer offload: partition the PLIO ports, and spec the PL compute kernels.

When a layer carries ``run_on='pl'`` the ExcisePLNodes pass cuts it out of the AIE graph
(passes/pl_excise.py), turning its two boundary tensors into extra PLIO ports. The array then has
more PLIO than the model's own inputs and outputs -- and the two roles are INTERLEAVED, not
appended:

    PLIO_ifm:  [0][1][2][3] [4][5]          PLIO_ofm:  [0][1] [2]
               \\___model__/ \\_cut_/                    \\_cut_/ \\model/
                  -> mm2s     <- PL kernel               -> PL     -> s2mm

  * this module  -- port ROLES + each PL compute kernel's spec (an IR/plan projection).
  * system_plan  -- nk=/sc= connectivity, the kernel list, the on-chip budget. It stays the single
                    authority for connectivity; it just consumes the partition computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# PL compute kernels live under pl/compute/; the data movers live under pl/benchmark|deployment/.
_COMPUTE_TEMPLATE_DIR = 'pl/compute'
# The generic AXIS wrapper around an hls4ml-generated op body (pl_hls4ml.py fills the firmware +
# template vars at write time). 'dummy_softmax' remains available for plumbing/bring-up tests.
_HLS4ML_KERNEL = 'hls4ml_kernel'
_DUMMY_KERNEL = 'dummy_softmax'

# CU names the data movers already own. A layer may not collide with one, or system.cfg would
# declare two different kernels under the same nk= name.
_RESERVED_CU_NAMES = frozenset({'mm2s', 's2mm', 'tick_gen', 'traffic_gen', 'ddr_pl_aie_datamover'})


@dataclass(frozen=True)
class PLKernelSpec:
    """One PL compute kernel standing in for an excised AIE layer."""

    name: str  # CU name == the layer name, e.g. 'softmax_0'
    source_layer: str
    cut_out_ports: List[int]  # PLIO_ofm indices -> kernel.s_in_*  (AIE -> PL)
    cut_in_ports: List[int]  # kernel.s_out_*   -> PLIO_ifm        (PL -> AIE)
    beats_per_iter: int  # AXIS beats ONE stream carries per iteration
    cpp_template: str
    cfg_template: str

    @property
    def n_in(self) -> int:
        return len(self.cut_out_ports)

    @property
    def n_out(self) -> int:
        return len(self.cut_in_ports)


@dataclass(frozen=True)
class BoundaryPlan:
    """PLIO ports partitioned into the model boundary (DDR movers) and the PL-kernel cuts."""

    model_ifm_ports: List[int]  # -> mm2s
    model_ofm_ports: List[int]  # -> s2mm
    model_input_tensor: str
    model_output_tensor: str
    kernels: List[PLKernelSpec]

    @property
    def has_cuts(self) -> bool:
        return bool(self.kernels)


def resolve_pl_offload(model_or_ctx, layout) -> BoundaryPlan:
    """Partition the PLIO ports and describe each PL compute kernel.

    ``layout`` is a :class:`simulation.IOLayout` built from the physical plan; its per-tensor port
    lists are already sorted by PLIO index. Port roles are ALWAYS derived from it, never assumed:
    a cut tensor does not land in source order (in tutorial_4 the cut takes ofm[0..1] while the
    real model output takes ofm[2]).

    With no cuts this degenerates to the classic single-input/single-output check, so an AIE-only
    hardware build behaves exactly as before.
    """
    ctx = get_backend_context(model_or_ctx)
    cuts = list(ctx.ir.logical.pl_cuts)

    cut_in_names = ctx.ir.logical.cut_in_tensor_names()  # PL -> AIE (extra graph INPUTS)
    cut_out_names = ctx.ir.logical.cut_out_tensor_names()  # AIE -> PL (extra graph OUTPUTS)

    model_inputs = [t for t in layout.inputs if t not in cut_in_names]
    model_outputs = [t for t in layout.outputs if t not in cut_out_names]
    if len(model_inputs) != 1 or len(model_outputs) != 1:
        raise RuntimeError(
            'system I/O plan supports exactly one graph input and one graph output tensor. After '
            f'setting aside {len(cuts)} PL cut(s), it found input(s)={model_inputs} and '
            f'output(s)={model_outputs}. Multiple model I/O tensors are not yet supported.'
        )

    return BoundaryPlan(
        model_ifm_ports=[p.port for p in layout.inputs[model_inputs[0]]],
        model_ofm_ports=[p.port for p in layout.outputs[model_outputs[0]]],
        model_input_tensor=model_inputs[0],
        model_output_tensor=model_outputs[0],
        kernels=[_kernel_spec(ctx, layout, cut) for cut in cuts],
    )


def _kernel_spec(ctx, layout, cut) -> PLKernelSpec:
    """Describe the PL kernel that replaces one excised layer."""
    out_ports = layout.outputs.get(cut.cut_out_tensor)  # AIE -> PL
    in_ports = layout.inputs.get(cut.cut_in_tensor)  # PL  -> AIE
    if not out_ports or not in_ports:
        raise RuntimeError(
            f'{cut.source_layer}: the cut tensors ({cut.cut_out_tensor!r} out, {cut.cut_in_tensor!r} '
            'in) did not materialize as PLIO ports. The physical plan and the recorded cut disagree.'
        )

    # The two port counts are resolved INDEPENDENTLY: cut_out follows the producer's output sharding
    # (dense_1's CAS_NUM) and cut_in follows the consumer's input sharding (dense_2's CAS_LENGTH).
    # They are both 2 in tutorial_4 only because the dims work out that way -- a differently shaped
    # consumer could want 4 input chains, giving 2 ports out and 4 back in. The kernel maps
    # s_in_k -> s_out_k one-to-one, so it cannot re-shard; reject rather than silently mis-wire.
    if len(out_ports) != len(in_ports):
        raise NotImplementedError(
            f'{cut.source_layer}: the cut leaves the array on {len(out_ports)} PLIO port(s) and '
            f're-enters on {len(in_ports)}. v1 maps s_in_k -> s_out_k one-to-one and cannot '
            're-partition across the cut.'
        )

    beats_out = _beats_per_iter(cut.source_layer, 'AIE->PL', ctx, out_ports)
    beats_in = _beats_per_iter(cut.source_layer, 'PL->AIE', ctx, in_ports)
    if beats_out != beats_in:
        # Caught here rather than in hw_emu, where a volume mismatch presents as an unexplained hang.
        raise NotImplementedError(
            f'{cut.source_layer}: the cut carries {beats_out} beat(s)/stream out of the array but '
            f'{beats_in} back in. The PL kernel would have to change the data volume, which an '
            'elementwise op cannot.'
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
        cut_out_ports=[p.port for p in out_ports],
        cut_in_ports=[p.port for p in in_ports],
        beats_per_iter=beats_out,
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
