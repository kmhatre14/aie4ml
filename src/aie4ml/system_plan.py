# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""System-level (PL data mover + host) emission planning.

The AIE array project is always emitted. When the user selects ``target='hardware'`` on the
frontend, the project additionally emits the PL data mover, the v++ linker connectivity, and
the XRT host program so the design can run on a board (not just simulate the array).
``emits_system`` is the single gate consulted by the writer and ``AIEModel`` for that opt-in.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# Bytes in one 512-bit DDR/AXI word -- the unit the PL data mover transfers.
# This is a transport constant (matches the kernel's ap_uint<512> m_axi word); it is
# independent of the element dtype (int8/int16/int32 just change how many elements fit
# per word, not the word size).
_DDR_WORD_BYTES = 64

# The data mover runs iterations in fixed-size groups, so the usable iteration count must
# be a whole multiple of this
_ITERATIONS_PER_GROUP = 8

# Fraction of the on-chip RAM pool the planner is allowed to spend on preload/stream buffers.
# The model assumes perfect packing, but HLS's adds some overhead
# Budgeting only 80% of the pool leaves headroom for the overhead
_PL_USABLE_FRACTION = 0.8


def _max_preloadable_iterations(
    ctx, pl_memory: str, n_ifm: int, n_ofm: int, ifm_per_stream: int, ofm_per_stream: int
) -> int:
    """Largest n_iter whose benchmark preload buffers fit the PL on-chip pool (URAM or BRAM,
    whichever PLMemory selects), rounded down to a whole group of _ITERATIONS_PER_GROUP.

    Counts BLOCKS, not bytes (see _onchip_blocks): the preload buffers store 512-bit words, so
    each per-stream bank is width-pinned and its depth (per_stream * n_iter) rounds up to the
    block depth. A byte budget would over-count and hand back an n_iter that then over-utilizes
    at v++ link. Benchmark-only: that mover stages every iteration on-chip (one buffer per
    stream, no ping-pong -> copies=1); the memory_stream movers stream instead.
    """
    avail_blocks, depth, width, _ = _pl_pool(ctx, pl_memory)
    if avail_blocks <= 0:
        raise RuntimeError('PL on-chip budget is unknown for this device (missing UltraRAM/BlockRAM block geometry).')

    def _blocks(n_iter: int) -> int:
        return _onchip_blocks(512, ifm_per_stream * n_iter, n_ifm, depth, width, copies=1) + _onchip_blocks(
            512, ofm_per_stream * n_iter, n_ofm, depth, width, copies=1
        )

    # Grow by whole groups while they still fit (block count is monotonic in n_iter).
    possible_iters = 0
    while _blocks(possible_iters + _ITERATIONS_PER_GROUP) <= avail_blocks:
        possible_iters += _ITERATIONS_PER_GROUP
    # Error if not even one group fits -- per-iteration footprint too large for this device.
    if possible_iters < _ITERATIONS_PER_GROUP:
        raise RuntimeError(
            f'PL on-chip pool ({avail_blocks} blocks) cannot hold one group of '
            f'{_ITERATIONS_PER_GROUP} preloaded iterations (needs {_blocks(_ITERATIONS_PER_GROUP)} '
            f'blocks); the per-iteration IO footprint is too large for this device.'
        )
    return possible_iters


# A 512-bit data-mover word is WIDTH-PINNED to ceil(512/WidthBits)=8 blocks no matter how shallow it is
# which is why a shallow ping-pong buffer can exhaust the pool by block count even when its byte footprint
# looks tiny. So the budgets below count blocks, not bytes.


def _pl_pool(ctx, pl_memory: str):
    """(usable_blocks, depth, width_bits, label) for the on-chip RAM pool PLMemory selects, read
    from the device catalog (aie_devices.json -> ctx.device)."""
    d = ctx.device
    if pl_memory == 'bram':
        blocks, depth, width, label = int(d.bram_blocks), int(d.bram_depth), int(d.bram_width_bits), 'BRAM'
    else:
        blocks, depth, width, label = int(d.uram_blocks), int(d.uram_depth), int(d.uram_width_bits), 'URAM'
    return int(blocks * _PL_USABLE_FRACTION), depth, width, label


def _onchip_blocks(
    word_bits: int, rows: int, n_banks: int, block_depth: int, block_width_bits: int, copies: int = 2
) -> int:
    """RAM blocks for `copies` ping-pong buffers, each `n_banks` independent banks of `rows`
    deep x `word_bits` wide. Width-pinned: ceil(word_bits/block_width_bits) blocks per bank,
    depth rounded up to block_depth."""
    width_blocks = math.ceil(int(word_bits) / int(block_width_bits))
    depth_blocks = math.ceil(int(rows) / int(block_depth))
    return int(copies) * int(n_banks) * width_blocks * depth_blocks


def _stream_buffer_blocks(ctx, pl_memory: str, ifm_banks: List, ofm_banks: List) -> int:
    """Total URAM/BRAM blocks the memory_stream ping-pong buffers occupy (mm2s + s2mm) in the
    pool PLMemory selects. Both ``ifm_banks`` and ``ofm_banks`` are ``(n_streams, depth)`` per-mover
    lists (one entry per input / output tensor): n_streams banks x depth rows each, so
    differently-shaped tensors cost their own depth."""
    _, depth, width, _ = _pl_pool(ctx, pl_memory)
    ifm_blocks = sum(_onchip_blocks(512, d, k, depth, width) for k, d in ifm_banks)
    ofm_blocks = sum(_onchip_blocks(512, d, k, depth, width) for k, d in ofm_banks)
    return ifm_blocks + ofm_blocks


def _check_memory_stream_fits(ctx, pl_memory: str, ifm_banks: List, ofm_banks: List) -> int:
    """If the memory_stream ping-pong buffers exceed the on-chip pool, counted in BLOCKS.
    ``ifm_banks`` / ``ofm_banks`` are ``(n_streams, per_stream)`` per input/output-tensor mover
    lists. Suggests the other pool when it would fit. Returns blocks."""
    avail, _, width, label = _pl_pool(ctx, pl_memory)
    needed = _stream_buffer_blocks(ctx, pl_memory, ifm_banks, ofm_banks)
    if avail and needed > avail:
        other = 'bram' if pl_memory == 'uram' else 'uram'
        o_avail, _, _, o_label = _pl_pool(ctx, other)
        o_needed = _stream_buffer_blocks(ctx, other, ifm_banks, ofm_banks)
        if o_avail and o_needed <= o_avail:
            hint = f" PLMemory='{other}' would fit ({o_needed}/{o_avail} {o_label}); try that."
        else:
            hint = ' Reduce the PLIO count (coarser slice) or target a larger device.'
        raise RuntimeError(
            f'memory_stream buffers need {needed} {label} blocks but only {avail} are available '
            f'(2 ping-pong x [{sum(k for k, _ in ifm_banks)} ifm + {sum(k for k, _ in ofm_banks)} ofm] '
            f'banks; each 512-bit word pins {math.ceil(512 / width)} blocks/bank).' + hint
        )
    return needed


# PL data-mover template locations. The benchmark mover lives under pl/benchmark/; the
# deployment movers (memory_stream / external_stream) live under pl/deployment/.
_BENCHMARK_KERNEL = 'ddr_pl_aie_datamover'
_BENCHMARK_TEMPLATE_DIR = 'pl/benchmark'
_DEPLOYMENT_TEMPLATE_DIR = 'pl/deployment'


def _ifm_mover_name(index: int) -> str:
    """CU name for the memory_stream input mover serving the ``index``-th graph input PLIO port.

    Port 0 keeps the historical bare name ``mm2s`` (so single-input designs are unchanged);
    later ports get ``mm2s_2``, ``mm2s_3``, ... . All movers render from the one parameterized
    ``mm2s.{cpp,cfg}.jinja`` (see :func:`_kernel_entry`'s ``template_base``), so any N is supported
    with no extra template files.
    """
    return 'mm2s_1' if index == 0 else f'mm2s_{index + 1}'


def _ofm_mover_name(index: int) -> str:
    """CU name for the memory_stream output mover serving the ``index``-th graph output tensor.

    Mirror of :func:`_ifm_mover_name` for the s2mm side: tensor 0 keeps the bare name ``s2mm``
    (so single-output designs are unchanged); later tensors get ``s2mm_2``, ``s2mm_3``, ... . All
    render from the one parameterized ``s2mm.{cpp,cfg}.jinja``.
    """
    return 's2mm_1' if index == 0 else f's2mm_{index + 1}'
# Hardware-emission vocabulary — the single source of truth both frontends validate against.
PL_TARGETS = ('aie', 'hardware')
PL_MEMORIES = ('uram', 'bram')
PL_DATA_MOVER_MODES = ('benchmark', 'memory_stream', 'external_stream')


def _one_of(cfg: Dict[str, Any], key: str, valid, default: str) -> None:
    value = str(cfg.get(key, default)).lower()
    if value not in valid:
        raise ValueError(f'AIEConfig.{key} must be one of {valid}, got {cfg.get(key)!r}.')
    cfg[key] = value


def normalize_pl_config(aie_config: Dict[str, Any]) -> Dict[str, Any]:
    _one_of(aie_config, 'Target', PL_TARGETS, 'aie')
    _one_of(aie_config, 'PLMemory', PL_MEMORIES, 'uram')
    _one_of(aie_config, 'PLDataMoverMode', PL_DATA_MOVER_MODES, 'benchmark')
    aie_config['EnablePLTiming'] = bool(aie_config.get('EnablePLTiming', False))
    return aie_config


def emits_system(model_or_ctx) -> bool:
    """Return True when PL + host code should be emitted (``target='hardware'``).

    Defaults to AIE-only emission (``target='aie'``) when the key is absent.
    """
    ctx = get_backend_context(model_or_ctx)
    return str(ctx.aie_config.get('Target', 'aie')).lower() == 'hardware'


def _single_io_feat(ports_map: Dict[str, Any], direction: str, batch: int):
    """Return (per-sample feature count, element bytes) for a single graph IO tensor.

    Supports exactly one graph input and one graph output (multiple graph I/O tensors are
    not yet supported).
    """
    tensors = list(ports_map)
    port0 = ports_map[tensors[0]][0]
    total = int(math.prod(port0.numpy_boundary_shape))
    if total % int(batch) != 0:
        raise RuntimeError(
            f'graph {direction} {tensors[0]!r}: element count {total} is not divisible by batch {batch}.'
        )
    return total // int(batch), int(port0.dtype.width) // 8


def _build_ifm_tensor_specs(inputs_map: Dict[str, Any], batch: int) -> List[Dict[str, int]]:
    """One spec per graph INPUT TENSOR, ordered by its first (lowest) PLIO port index -- so spec j
    is served by input mover j.

    Each tensor is delivered by ONE mm2s CU that feeds ALL of the tensor's PLIO streams: a single
    sharded input therefore stays a single mover (matching the original single-input design), while
    N distinct input tensors become N movers. Because the per-stream depth is derived per tensor,
    differently-shaped inputs get differently-sized (non byte-identical) movers.

    Fields: ``port_base`` (first PLIO_ifm port the tensor owns), ``n_streams`` (contiguous PLIO
    ports it spans), ``ifm_per_stream`` (512-bit ping-pong depth per stream), ``size_words``
    (32-bit words/iteration for the whole tensor -- the host mover's ``size_in``).
    """
    specs: List[Dict[str, int]] = []
    for tname, ports in inputs_map.items():
        p0 = ports[0]
        elem_bytes = int(p0.dtype.width) // 8
        total = int(math.prod(p0.numpy_boundary_shape))
        if total % int(batch) != 0:
            raise RuntimeError(f'graph input {tname!r}: element count {total} not divisible by batch {batch}.')
        feat = total // int(batch)
        n_streams = len(ports)
        specs.append(
            {
                'tensor': tname,
                'port_base': min(int(p.port) for p in ports),
                'n_streams': n_streams,
                'ifm_per_stream': _stream_words_512(batch, feat, elem_bytes, n_streams, 'input'),
                'size_words': (int(batch) * feat * elem_bytes) // 4,
            }
        )
    specs.sort(key=lambda s: s['port_base'])
    return specs


def _build_ofm_tensor_specs(outputs_map: Dict[str, Any], batch: int) -> List[Dict[str, int]]:
    """One spec per graph OUTPUT TENSOR, ordered by its first (lowest) PLIO port index -- so spec j
    is served by output mover j. Mirror of :func:`_build_ifm_tensor_specs` for the s2mm side.

    Each output tensor is drained by ONE s2mm CU that receives ALL of the tensor's PLIO_ofm
    streams; a single sharded output stays one mover, N distinct output tensors become N movers.

    Fields: ``port_base`` (first PLIO_ofm port), ``n_streams`` (contiguous PLIO ports it spans),
    ``ofm_per_stream`` (512-bit ping-pong depth per stream), ``size_words`` (32-bit words/iteration
    for the whole tensor -- the host mover's ``size_out``).
    """
    specs: List[Dict[str, int]] = []
    for tname, ports in outputs_map.items():
        p0 = ports[0]
        elem_bytes = int(p0.dtype.width) // 8
        total = int(math.prod(p0.numpy_boundary_shape))
        if total % int(batch) != 0:
            raise RuntimeError(f'graph output {tname!r}: element count {total} not divisible by batch {batch}.')
        feat = total // int(batch)
        n_streams = len(ports)
        specs.append(
            {
                'tensor': tname,
                'port_base': min(int(p.port) for p in ports),
                'n_streams': n_streams,
                'ofm_per_stream': _stream_words_512(batch, feat, elem_bytes, n_streams, 'output'),
                'size_words': (int(batch) * feat * elem_bytes) // 4,
            }
        )
    specs.sort(key=lambda s: s['port_base'])
    return specs


def _stream_words_512(batch: int, feat: int, elem_bytes: int, n_streams: int, direction: str) -> int:
    """512-bit words per stream per iteration; requires 512-bit + per-stream alignment."""
    total_bytes = int(batch) * int(feat) * int(elem_bytes)
    if total_bytes % (_DDR_WORD_BYTES * int(n_streams)) != 0:
        raise NotImplementedError(
            f'graph {direction}: {total_bytes} bytes is not a multiple of '
            f'{_DDR_WORD_BYTES} B/word * {n_streams} stream(s); 512-bit/per-stream padding is not yet supported.'
        )
    return total_bytes // (_DDR_WORD_BYTES * int(n_streams))


def _kernel_entry(name: str, template_dir: str, context: Dict[str, Any] | None = None, template_base: str | None = None) -> Dict[str, Any]:
    """One PL kernel for the writer (which templates to render) and the Makefile (.xo).

    ``context`` holds per-kernel template overrides merged over the shared ``system_io`` bag
    at render time (e.g. an mm2s mover's own ``n_streams`` / ``kernel_name``). ``template_base``
    lets several CUs share ONE template file: the N input movers all render the single
    parameterized ``mm2s.{cpp,cfg}.jinja`` to distinct ``mm2s`` / ``mm2s_2`` / ``mm2s_3`` ...
    outputs. Defaults to ``name`` (one template file per kernel).
    """
    base = template_base or name
    return {
        'name': name,
        'cpp_template': f'{template_dir}/{base}.cpp.jinja',
        'cfg_template': f'{template_dir}/{base}.cfg.jinja',
        'context': dict(context or {}),
    }


def build_pl_plan(
    mode: str, ifm_tensors: List[Dict[str, int]], ofm_tensors: List[Dict[str, int]], enable_pl_timing: bool
) -> Dict[str, Any]:
    """Describe the PL data path for a given ``PLDataMoverMode``: which kernels to emit,
    the v++ linker connectivity (``nk=`` / ``sc=``), and the host's timing wiring.

    ``ifm_tensors`` / ``ofm_tensors`` are the per-graph-tensor spec lists from
    :func:`_build_ifm_tensor_specs` / :func:`_build_ofm_tensor_specs` (``port_base`` / ``n_streams``
    plus ``ifm_per_stream`` or ``ofm_per_stream``), one entry per input / output tensor.
    memory_stream emits one mm2s CU per input tensor and one s2mm CU per output tensor; the total
    input/output PLIO counts are ``sum(n_streams)``.

    This is the single source of truth consumed by:
      * the writer            -> ``plan['kernels']`` (which .cpp/.cfg templates to render),
      * ``Makefile.jinja``    -> kernel ``.xo`` list + per-kernel build,
      * ``system.cfg.jinja``  -> ``plan['nk']`` / ``plan['sc']`` connectivity,
      * ``host.cpp.jinja``    -> ``plan['host']`` (timing CU + ``cycles_*`` register names).

    Modes:
      ``benchmark``     single combined ``ddr_pl_aie_datamover`` CU (today's design);
      ``memory_stream`` one ``mm2s`` CU per input tensor + one ``s2mm`` CU, double-buffered, plus a
                        shared ``tick_gen`` timer CU when PL timing is on (separate CUs have
                        independent ``ap_start``, so one timer gives a common time base);
      ``external_stream`` on-chip HLS ``traffic_gen`` source -> AIE -> ``s2mm`` (a
                        synthesizable stand-in for an external AXI producer); PL timing off.
    """
    mode = str(mode).lower()
    n_ifm = sum(int(t['n_streams']) for t in ifm_tensors)  # total input PLIO ports
    n_ofm = sum(int(t['n_streams']) for t in ofm_tensors)  # total output PLIO ports

    if mode == 'benchmark':
        # Benchmark is a cycle-accurate measurement harness -- the PL timers are its
        # whole point, so force them ON regardless of EnablePLTiming.
        if not enable_pl_timing:
            print("[aie4ml] PLDataMoverMode='benchmark' forces PL timers ON " '(overriding EnablePLTiming=False).')
            enable_pl_timing = True
        name = _BENCHMARK_KERNEL
        sc = [f'{name}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{name}.s_in_{s}' for s in range(n_ofm)]
        # The combined CU owns tick_gen internally; the host reads its 7 cycles_* regs.
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': [_kernel_entry(name, _BENCHMARK_TEMPLATE_DIR)],
            'nk': [f'{name}:1:{name}'],
            'sc': sc,
            'host': {
                'timing_kernel': name,
                'cycles': [
                    'preload_done',
                    'first_send',
                    'last_send',
                    'first_recv',
                    'last_recv',
                    'compute_done',
                    'total',
                ],
            },
        }

    if mode == 'memory_stream':
        # One mm2s CU per INPUT tensor and one s2mm CU per OUTPUT tensor. Each mover feeds/drains
        # ALL of its tensor's PLIO streams (N_*_STREAMS = the tensor's port count) and owns its own
        # DDR buffer, so a single sharded tensor stays ONE mover (== the original single-in/out
        # design) and N distinct tensors -> N movers. All render from the one parameterized
        # mm2s/s2mm.{cpp,cfg}.jinja (template_base), differing only by kernel_name / n_streams /
        # {ifm,ofm}_per_stream; mover j owns the contiguous PLIO range [port_base .. +n_streams).
        ifm_movers = [_ifm_mover_name(j) for j in range(len(ifm_tensors))]
        ofm_movers = [_ofm_mover_name(j) for j in range(len(ofm_tensors))]
        # tick_gen can be fed by exactly ONE input mover (send events) and ONE output mover
        # (recv/done events). If either side has >1 mover, disable PL timing (host-side timing
        # instead), mirroring external_stream.
        if enable_pl_timing and (len(ifm_movers) > 1 or len(ofm_movers) > 1):
            print(
                f"[aie4ml] PLDataMoverMode='memory_stream' with {len(ifm_movers)} input / "
                f'{len(ofm_movers)} output movers has no PL timers (tick_gen can own only one mover '
                'per side); using host-side timing (overriding EnablePLTiming=True).'
            )
            enable_pl_timing = False
        kernels = []
        nk = []
        sc = []
        for name, spec in zip(ifm_movers, ifm_tensors):
            k = int(spec['n_streams'])
            base = int(spec['port_base'])
            kernels.append(
                _kernel_entry(
                    name,
                    _DEPLOYMENT_TEMPLATE_DIR,
                    {'n_streams': k, 'kernel_name': name, 'ifm_per_stream': int(spec['ifm_per_stream'])},
                    template_base='mm2s',
                )
            )
            nk.append(f'{name}:1:{name}')
            sc += [f'{name}.s_out_{i}:ai_engine_0.PLIO_ifm_{base + i}' for i in range(k)]
        for name, spec in zip(ofm_movers, ofm_tensors):
            k = int(spec['n_streams'])
            base = int(spec['port_base'])
            kernels.append(
                _kernel_entry(
                    name,
                    _DEPLOYMENT_TEMPLATE_DIR,
                    {'n_streams': k, 'kernel_name': name, 'ofm_per_stream': int(spec['ofm_per_stream'])},
                    template_base='s2mm',
                )
            )
            nk.append(f'{name}:1:{name}')
            sc += [f'ai_engine_0.PLIO_ofm_{base + i}:{name}.s_in_{i}' for i in range(k)]
        # Host mover info: name + stream count (the host launch emits one nullptr per stream port).
        ifm_info = [{'name': n, 'n_streams': int(s['n_streams'])} for n, s in zip(ifm_movers, ifm_tensors)]
        ofm_info = [{'name': n, 'n_streams': int(s['n_streams'])} for n, s in zip(ofm_movers, ofm_tensors)]
        host = {'timing_kernel': None, 'cycles': [], 'ifm_movers': ifm_info, 'ofm_movers': ofm_info}
        if enable_pl_timing:
            ifm_k = ifm_movers[0]  # single input tensor here: its mover feeds tick_gen (as original)
            ofm_k = ofm_movers[0]  # single output tensor here: its mover feeds tick_gen (as original)
            timer = 'tick_gen'
            kernels.append(_kernel_entry(timer, _DEPLOYMENT_TEMPLATE_DIR))
            nk.append(f'{timer}:1:{timer}')
            # PL-to-PL event pulses give tick_gen a single time base across the CUs.
            sc += [
                f'{ifm_k}.ev_first_send:{timer}.ev_first_send',
                f'{ifm_k}.ev_last_send:{timer}.ev_last_send',
                f'{ofm_k}.ev_first_recv:{timer}.ev_first_recv',
                f'{ofm_k}.ev_last_recv:{timer}.ev_last_recv',
                f'{ofm_k}.ev_done:{timer}.ev_done',
            ]
            host = {
                'timing_kernel': timer,
                'cycles': ['first_send', 'last_send', 'first_recv', 'last_recv', 'total'],
                'ifm_movers': ifm_info,
                'ofm_movers': ofm_info,
            }
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': kernels,
            'nk': nk,
            'sc': sc,
            'host': host,
        }

    if mode == 'external_stream':
        # The DDR->PLIO input mover (mm2s) is replaced by an ON-CHIP HLS traffic_gen
        # The output side is the unchanged s2mm mover so the host still reads the
        # result back / golden-checks it. On a deployed board,
        # On read deployment the traffic_gen is swapped for the user's producer
        # IP wired the same way (sc=<producer>.M_AXIS:ai_engine_0.PLIO_ifm_*).
        #
        # traffic_gen emits no timing event pulses, so there is no shared time base -- PL timing
        # is unavailable in this mode; the host measures wall-clock latency instead.
        if enable_pl_timing:
            print(
                "[aie4ml] PLDataMoverMode='external_stream' has no PL timers "
                '(the traffic_gen source emits no events); using host-side timing '
                '(overriding EnablePLTiming=True).'
            )
            enable_pl_timing = False
        src_k, ofm_k = 'traffic_gen', 's2mm'
        # external_stream has a single output tensor (guarded); its one s2mm drains all n_ofm
        # streams. The parameterized s2mm template needs kernel_name/n_streams; ofm_per_stream
        # falls back to the shared system_io value (not overridden here).
        kernels = [
            _kernel_entry(src_k, _DEPLOYMENT_TEMPLATE_DIR),
            _kernel_entry(ofm_k, _DEPLOYMENT_TEMPLATE_DIR, {'n_streams': n_ofm, 'kernel_name': ofm_k}, template_base='s2mm'),
        ]
        nk = [f'{src_k}:1:{src_k}', f'{ofm_k}:1:{ofm_k}']
        sc = [f'{src_k}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{ofm_k}.s_in_{s}' for s in range(n_ofm)]
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': kernels,
            'nk': nk,
            'sc': sc,
            'host': {'timing_kernel': None, 'cycles': []},
        }
    raise ValueError(f'unknown PLDataMoverMode {mode!r}.')


def build_system_io(model_or_ctx) -> Dict[str, Any]:
    """Project the resolved IR + physical plan into the flat variable bag the PL/host
    templates in ``templates/system/`` consume.

    Reuses :func:`aie4ml.simulation.build_io_layout` for per-PLIO-port boundary/dtype data and
    the dense/matmul execution entries for per-layer RTP (weight/bias) loading. Scope: N graph
    input tensors (memory_stream -- one mm2s mover each), single graph output, 512-bit-aligned sizes.
    """
    from .simulation import build_io_layout

    ctx = get_backend_context(model_or_ctx)
    plan = ctx.ir.physical.plan
    if 'graph_input_count' not in plan or 'buffers' not in plan:
        raise RuntimeError('build_system_io requires a materialized physical plan; run the pipeline first.')

    layout = build_io_layout(ctx)
    n_ifm = int(plan['graph_input_count'])
    n_ofm = int(plan['graph_output_count'])
    batch = int(ctx.aie_config['BatchSize'])
    pl_data_mover_mode = str(ctx.aie_config.get('PLDataMoverMode', 'benchmark')).lower()
    # Multiple graph input OR output tensors are supported only by memory_stream (each gets its own
    # mm2s / s2mm mover); benchmark/external assume a single tensor on each side.
    if len(layout.inputs) > 1 and pl_data_mover_mode != 'memory_stream':
        raise RuntimeError(
            f'multiple graph input tensors ({len(layout.inputs)}) require '
            f"PLDataMoverMode='memory_stream' (each input needs its own mm2s mover); "
            f'got {pl_data_mover_mode!r}.'
        )
    if len(layout.outputs) > 1 and pl_data_mover_mode != 'memory_stream':
        raise RuntimeError(
            f'multiple graph output tensors ({len(layout.outputs)}) require '
            f"PLDataMoverMode='memory_stream' (each output needs its own s2mm mover); "
            f'got {pl_data_mover_mode!r}.'
        )

    # Per-tensor movers, ordered by first PLIO port (one mm2s CU per input tensor, one s2mm CU per
    # output tensor -- a single sharded tensor stays one mover). (n_streams, per_stream) per tensor
    # drives the on-chip budget below and build_pl_plan's kernel contexts.
    ifm_tensor_specs = _build_ifm_tensor_specs(layout.inputs, batch)
    ofm_tensor_specs = _build_ofm_tensor_specs(layout.outputs, batch)
    ifm_banks = [(s['n_streams'], s['ifm_per_stream']) for s in ifm_tensor_specs]
    ofm_banks = [(s['n_streams'], s['ofm_per_stream']) for s in ofm_tensor_specs]
    out_feat, out_bytes = _single_io_feat(layout.outputs, 'output', batch)
    # Aggregate input feature count / element bytes
    in_feat, _ = _single_io_feat(layout.inputs, 'input', batch)
    # Per-PLIO-tile feature slices come from the first graph boundary port of each direction
    gin_port = next(iter(layout.inputs.values()))[0]
    gout_port = next(iter(layout.outputs.values()))[0]
    in_feat_slice = gin_port.tiling_dimension[gin_port.slice_dimension]
    out_feat_slice = gout_port.tiling_dimension[gout_port.slice_dimension]

    # Per-layer RTP artifacts (weights/bias/...). layer_index counts every executed node
    # (incl. param-less activations) so the RTP port suffix matches the app.cpp graph index.
    layers = []
    layer_index = 0
    for node in ctx.ir.logical:
        inst = ctx.ir.execution.get(node.name)
        if inst is None:
            continue
        layer_index += 1
        artifacts = inst.variant.get_artifacts(inst)
        if not artifacts:
            continue  # param-less op (e.g. activation) -- no kernel-resident RTP
        parallelism = getattr(inst.config, 'parallelism', None)
        if parallelism is None:
            raise RuntimeError(f'{node.name}: RTP-bearing layer has no parallelism config.')
        inst_name = sanitize_identifier(node.name)
        layers.append(
            {
                'inst_name': inst_name,
                'cas_num': int(parallelism.cas_num),
                'cas_length': int(parallelism.cas_length),
                'artifacts': [
                    {
                        'name': a['name'],
                        'kind': a['kind'],  # '1d' | '2d' -> RTP loop shape
                        'header': a['filename'],  # generated header to #include
                        'prefix': f"{a['name']}_{inst_name}",  # C symbol base (matches header)
                        'port': f"{a['port']}{layer_index}",  # ADF RTP port (name + layer idx, == app.cpp Lidx)
                    }
                    for a in artifacts
                ],
            }
        )
    # Top-level cas_* describe the graph-output-producing (last weight) layer.
    cas_num = layers[-1]['cas_num'] if layers else 1
    cas_length = layers[-1]['cas_length'] if layers else 1
    # Single-output uneven-shard guard (multi-output is memory_stream-only and validates each
    # tensor's alignment inside _build_ofm_tensor_specs, so this scalar check applies to 1 output).
    if len(ofm_tensor_specs) == 1 and out_feat % cas_num != 0:
        raise NotImplementedError(
            f'out_feat {out_feat} not divisible by cas_num {cas_num}; uneven output shard is not yet supported.'
        )
    # Scalar depths for the single combined benchmark mover / comments (benchmark is single-in/out
    # by the guard); memory_stream sizes each mover per-tensor via its kernel context.
    ifm_per_stream = max((s['ifm_per_stream'] for s in ifm_tensor_specs), default=0)
    ofm_per_stream = max((s['ofm_per_stream'] for s in ofm_tensor_specs), default=0)
    iterations = int(ctx.aie_config['Iterations'])

    # On-chip pool selection -- common to every mover (all bind_storage to URAM or BRAM).
    pl_memory = str(ctx.aie_config.get('PLMemory', 'uram')).lower()
    pl_mem_impl = 'BRAM' if pl_memory == 'bram' else 'URAM'

    # One mm2s CU per input tensor + one s2mm CU per output tensor; build_pl_plan bakes each mover's
    # n_streams / {ifm,ofm}_per_stream into its kernel context directly (no post-hoc injection here).
    pl_plan = build_pl_plan(
        pl_data_mover_mode, ifm_tensor_specs, ofm_tensor_specs, bool(ctx.aie_config.get('EnablePLTiming', True))
    )
    enable_pl_timing = pl_plan['enable_pl_timing']

    # On-chip budget is mode-specific, but BOTH count in BLOCKS -- the buffers are 512-bit
    # words, so each per-stream bank is width-pinned to 8 URAM/BRAM blocks regardless of depth
    # (a byte budget badly under-counts):
    #   benchmark        preloads ALL n_iter on-chip -> cap n_iter to what the pool holds.
    #   memory_stream    fixed 2-deep ping-pong buffers (n_iter unbounded) -> just verify they fit.
    #   external_stream  like memory_stream but the input side is the on-chip traffic_gen, which
    #                    holds NO ping-pong buffer -> only the s2mm (ofm) buffers count.
    if pl_data_mover_mode == 'benchmark':
        max_n_iter = _max_preloadable_iterations(ctx, pl_memory, n_ifm, n_ofm, ifm_per_stream, ofm_per_stream)
    elif pl_data_mover_mode == 'external_stream':
        # traffic_gen streams on the fly (no input buffer): no ifm banks so only the s2mm (ofm)
        # ping-pong is charged.
        _check_memory_stream_fits(ctx, pl_memory, [], ofm_banks)
        max_n_iter = iterations
    else:
        # memory_stream: not preload-capped; fail early if the per-tensor ping-pong won't fit the pool.
        _check_memory_stream_fits(ctx, pl_memory, ifm_banks, ofm_banks)
        max_n_iter = iterations
    return {
        'project_name': ctx.project_config.project_name,
        'platform': ctx.device.platform,
        'graph_name': 'dut',
        'pl_freq_hz': float(ctx.aie_config['PLClockFreqMHz']) * 1e6,
        'plio_width_bits': int(ctx.device.plio_width_bits),
        'pl_mem_impl': pl_mem_impl,
        'enable_pl_timing': enable_pl_timing,
        'n_ifm': n_ifm,
        'n_ofm': n_ofm,
        'batch': batch,
        'iterations': iterations,
        'max_n_iter': max_n_iter,
        'in_feat': in_feat,
        'out_feat': out_feat,
        'in_feat_slice': in_feat_slice,
        'out_feat_slice': out_feat_slice,
        'cas_num': cas_num,
        'cas_length': cas_length,
        'ifm_per_stream': ifm_per_stream,
        'ofm_per_stream': ofm_per_stream,
        'layers': layers,
        'pl_data_mover_mode': pl_data_mover_mode,
        'pl_plan': pl_plan,
    }


# ---------------------------------------------------------------------------
# Host data.h generation (DDR-packed input) — target='hardware'
#
# The PL data movers (templates/firmware/pl/benchmark/ or pl/deployment/) move whole 512-bit
# DDR words and round-robin stripe them across the N per-direction PLIO streams
# (word i -> stream i % N). The host (host.cpp) replays this packed input each iteration.
# This produces `host/data.h` with the packed input and the IO word sizes.
# ---------------------------------------------------------------------------


def _pack_ports_to_ddr(port_tiles: List[np.ndarray], n_streams: int) -> np.ndarray:
    """Round-robin pack per-stream tiles into the data mover's 512-bit DDR layout.

    Packs each tile's RAW STORAGE BYTES (dtype-agnostic -- the PL data mover moves whole
    512-bit words and is type-agnostic; int8/int16/int32 just change how many elements fit
    per word). Returns a little-endian uint32 array (the host buffer element type), with
    contiguous DDR word i owned by stream i % n_streams. Assumes little-endian byte order
    (native on x86/Versal), consistent with the kernel's word reads.
    """
    word_blocks = []
    for tile in port_tiles:
        raw = np.frombuffer(np.ascontiguousarray(tile).tobytes(), dtype=np.uint8)
        if raw.size % _DDR_WORD_BYTES != 0:
            raise NotImplementedError(
                f'stream tile of {raw.size} B is not a multiple of {_DDR_WORD_BYTES} B; '
                f'512-bit padding is not yet supported.'
            )
        word_blocks.append(raw.reshape(-1, _DDR_WORD_BYTES))
    if len(word_blocks) != int(n_streams):
        raise RuntimeError(f'expected {n_streams} stream tiles, got {len(word_blocks)}.')
    words = word_blocks[0].shape[0]
    if any(block.shape[0] != words for block in word_blocks):
        raise NotImplementedError('uneven per-stream word counts; relay/padding is not yet supported.')

    mem = np.empty((words * int(n_streams), _DDR_WORD_BYTES), dtype=np.uint8)
    for stream, block in enumerate(word_blocks):
        mem[stream :: int(n_streams)] = block
    return np.frombuffer(mem.tobytes(), dtype='<u4')


def _check_storage_width(port, tile) -> None:
    """Fail hard if the prepared element width doesn't match the port's byte-aligned dtype.

    The packing is dtype-aware via raw bytes, but only for byte-aligned formats whose
    prepared NumPy dtype matches the boundary width (int8/int16/int32). Sub-byte widths or
    formats whose storage differs (e.g. bfloat16 prepared as float32) are rejected rather
    than silently corrupted.
    """
    if int(port.dtype.width) % 8 != 0:
        raise NotImplementedError(
            f'graph IO dtype {port.dtype.format!r} ({port.dtype.width}-bit) is not byte-aligned; '
            f'sub-byte PLIO packing is not supported.'
        )
    want = int(port.dtype.width) // 8
    got = int(tile.dtype.itemsize)
    if got != want:
        raise NotImplementedError(
            f'graph input {port.tensor!r} port {port.port}: prepared element is {got} B but boundary '
            f'dtype {port.dtype.format!r} is {want} B; dtype-exact packing for this format is unsupported.'
        )


def pack_host_data(model_or_ctx, X=None):
    """Return (ifm_packed, ifm_packed_tensors, ofm_size_words) for one iteration, in the data mover
    DDR layout. ``ifm_packed`` is the combined buffer (benchmark's single mover); ``ifm_packed_tensors``
    is one round-robin buffer per graph INPUT TENSOR (memory_stream's per-tensor mm2s movers each
    read one), ordered by the tensor's first PLIO port.

    Packs the quantized graph input into the DDR layout the host replays, and reports the
    output size (in 32-bit words) the host needs to size its output buffer. Both are
    graph-agnostic (no forward pass). X defaults to a deterministic pseudo-random input.

    dtype-aware: ``prepare_inputs`` quantizes X to the port's storage NumPy dtype
    (``dtype_for_precision(width, signed)`` for ints, float32 for floats) and the raw bytes
    of that are packed -- the same storage-dtype contract the weights path uses. n-D
    boundaries are handled via the full ``numpy_boundary_shape``.
    """
    from .simulation import _extract_port_tile, build_io_layout, prepare_inputs

    ctx = get_backend_context(model_or_ctx)
    layout = build_io_layout(ctx)
    out_port0 = layout.outputs[next(iter(layout.outputs))][0]

    # One X per graph input tensor. Default: deterministic pseudo-random per tensor (its own seed
    # so distinct inputs get distinct data). A user-supplied X may be a single array (one input)
    # or a {tensor_name: array} dict (prepare_inputs handles both).
    if X is None:
        X = {
            tname: np.random.default_rng(seed).random(ports[0].numpy_boundary_shape, dtype=np.float64) * 2.0 - 1.0
            for seed, (tname, ports) in enumerate(layout.inputs.items())
        }
    prepared = prepare_inputs(layout, X, iterations=1, quantize=True)  # {tensor: (1, *boundary)}

    # One DDR buffer per INPUT TENSOR (== one per mm2s mover), ordered by the tensor's first PLIO
    # port. Each tensor's own PLIO-port tiles are round-robin interleaved into its buffer, exactly
    # as the original single-input mover expects; a single sharded input yields exactly one buffer.
    tensor_order = sorted(layout.inputs.items(), key=lambda kv: min(int(p.port) for p in kv[1]))
    ifm_packed_tensors = []
    for tname, ports in tensor_order:
        tiles = []
        for p in ports:
            tile = _extract_port_tile(prepared[tname], p)[0]  # this port's slice, storage dtype, n-D
            _check_storage_width(p, tile)
            tiles.append(tile)
        ifm_packed_tensors.append(_pack_ports_to_ddr(tiles, len(tiles)))

    # Combined buffer for benchmark's single mover (single input by the guard): its one tensor's
    # buffer. Multi-tensor is memory_stream-only, so concatenate purely so data.h's ifm_packed[]
    # exists (benchmark never reads it there).
    if len(ifm_packed_tensors) == 1:
        ifm_packed = ifm_packed_tensors[0]
    else:
        ifm_packed = np.concatenate(ifm_packed_tensors) if ifm_packed_tensors else np.zeros(0, dtype='<u4')

    # Per-OUTPUT-TENSOR sizes (32-bit words), ordered by first PLIO port -- one per s2mm mover.
    # The device produces the output data, so the host only needs each buffer's size to allocate it.
    ofm_order = sorted(layout.outputs.items(), key=lambda kv: min(int(p.port) for p in kv[1]))
    ofm_size_tensors = []
    for tname, ports in ofm_order:
        p0 = ports[0]
        obytes = int(np.prod(p0.numpy_boundary_shape)) * (int(p0.dtype.width) // 8)
        if obytes % 4 != 0:
            raise NotImplementedError(f'graph output {tname!r} is {obytes} B, not a multiple of 4 B (uint32 host buffer).')
        ofm_size_tensors.append(obytes // 4)
    ofm_size_words = sum(ofm_size_tensors)  # total (== the single tensor's size for benchmark/external)
    return ifm_packed, ifm_packed_tensors, ofm_size_words, ofm_size_tensors


def host_data_context(model_or_ctx, X=None) -> Dict[str, Any]:
    """Prepare the ``host/data.h`` template context: the DDR-packed graph input and
    the IO word sizes the host needs.

    Python only prepares the values (masked to uint32); the writer renders
    ``templates/firmware/host/data.h.jinja``.
    """
    ifm_packed, ifm_packed_tensors, ofm_size_words, ofm_size_tensors = pack_host_data(model_or_ctx, X)
    # One entry per input tensor (== per mm2s mover), index-aligned with pl_plan.host.ifm_movers.
    ifm_ports = [
        {'index': i, 'data': [int(v) & 0xFFFFFFFF for v in arr], 'size_words': int(len(arr))}
        for i, arr in enumerate(ifm_packed_tensors)
    ]
    # One size per output tensor (== per s2mm mover), index-aligned with pl_plan.host.ofm_movers.
    ofm_ports = [{'index': i, 'size_words': int(w)} for i, w in enumerate(ofm_size_tensors)]
    return {
        'ifm_packed': [int(v) & 0xFFFFFFFF for v in ifm_packed],
        'ifm_size_words': int(len(ifm_packed)),
        # Per-input-tensor buffers (memory_stream's per-tensor mm2s movers each read one).
        'ifm_ports': ifm_ports,
        # Per-output-tensor sizes (memory_stream's per-tensor s2mm movers each write one).
        'ofm_ports': ofm_ports,
        'ofm_size_words': int(ofm_size_words),
    }
