# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Drive a Vitis ``hw_emu`` (QEMU) run non-interactively and collect the
performance report the generated host program already prints.

The hardware-emulation flow boots a full QEMU/PetaLinux image, autologins to a
root shell, and expects the user to launch the host by hand (``make run_emu``
then ``./host.exe a.xclbin <iters>``). This module automates that handshake with
:mod:`pexpect`: boot -> wait for shell -> run host -> wait for the host's
completion sentinel -> parse the printed perf -> power off.

It does *not* read back the OFM tensor; only the timing/throughput numbers the
host already emits on the console are captured and returned as a nested dict.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    import pexpect
except ImportError as exc:  # pragma: no cover - exercised only without pexpect
    pexpect = None
    _PEXPECT_IMPORT_ERROR: Optional[Exception] = exc
else:
    _PEXPECT_IMPORT_ERROR = None

_LAUNCH_SCRIPT = 'launch_hw_emu.sh'
_HOST_EXE = 'host.exe'

# Final line printed by the generated host (host.cpp.jinja) once a run finishes.
# Emitted *after* the full performance block, so by the time we match it the
# transcript already contains everything worth parsing.
_DONE_RE = r'ran\s+\d+\s+iteration\(s\)'

# Signals that we're past the bootloader and into the kernel/userspace -- safe to
# start nudging the console without risking interrupting U-Boot autoboot.
_BOOTED_RE = r'(?i)(Starting kernel|Linux version|systemd\[1\]|Reached target|login:)'

# (key, pattern) per section; each pattern captures a single number.
_PERF_PATTERNS: Dict[str, list] = {
    'host': [
        ('total_ms', r'Total time \(host\)\s*:\s*([-\d.]+)\s*ms'),
        ('latency_ms', r'Latency \(host avg\)\s*:\s*([-\d.]+)\s*ms/iter'),
        ('throughput_ips', r'Throughput\s*:\s*([-\d.]+)\s*inferences/s'),
        ('bandwidth_gbs', r'Bandwidth\s*:\s*([-\d.]+)\s*GB/s'),
    ],
    'pl': [
        ('clock_mhz', r'PL clock\s*:\s*([-\d.]+)\s*MHz'),
        ('total_cyc', r'datamover total wall\s*:\s*([-\d.]+)\s*cyc'),
        ('total_ms', r'datamover total wall\s*:[^()]*\(\s*([-\d.]+)\s*ms\)'),
        ('preload_cyc', r'Phase1 preload\s*:\s*([-\d.]+)\s*cyc'),
        ('preload_us_total', r'Phase1 preload\s*:[^()]*\(\s*([-\d.]+)\s*us total'),
        ('compute_cyc', r'Phase2 compute\s*:\s*([-\d.]+)\s*cyc'),
        ('compute_us_total', r'Phase2 compute\s*:[^()]*\(\s*([-\d.]+)\s*us total'),
        ('postwrite_cyc', r'Phase3 postwrite\s*:\s*([-\d.]+)\s*cyc'),
        ('postwrite_us_total', r'Phase3 postwrite\s*:[^()]*\(\s*([-\d.]+)\s*us total'),
        ('aie_compute_us_iter', r'AIE compute rate \(Phase2 / iter\)\s*:\s*([-\d.]+)\s*us/iter'),
    ],
    'iter0': [
        ('send_phase_us', r'send_phase\s*:[^()]*\(\s*([-\d.]+)\s*us\)'),
        ('aie_gap_us', r'aie_gap\s*:[^()]*\(\s*([-\d.]+)\s*us\)'),
        ('recv_phase_us', r'recv_phase\s*:[^()]*\(\s*([-\d.]+)\s*us\)'),
        ('roundtrip_us', r'PL->AIE->PL RT\s*:[^()]*\(\s*([-\d.]+)\s*us\)'),
    ],
}


def _coerce(value: str):
    return float(value) if ('.' in value or 'e' in value.lower()) else int(value)


def parse_perf(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract the host/PL performance numbers from a captured console log.

    In addition to the raw ``host``/``pl``/``iter0`` sections, a ``summary`` dict
    surfaces the two headline latencies with friendly names:
      * ``iteration_sample_latency_us`` -- steady-state AIE compute rate, from
        ``AIE compute rate (Phase2 / iter): X us/iter``;
      * ``single_iter_latency_us`` -- the iter-0 ``PL->AIE->PL RT`` round trip.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for section, patterns in _PERF_PATTERNS.items():
        section_vals: Dict[str, Any] = {}
        for key, pattern in patterns:
            match = re.search(pattern, text)
            if match:
                section_vals[key] = _coerce(match.group(1))
        if section_vals:
            out[section] = section_vals

    summary: Dict[str, Any] = {}
    if out.get('pl', {}).get('aie_compute_us_iter') is not None:
        summary['iteration_sample_latency_us'] = out['pl']['aie_compute_us_iter']
    if out.get('iter0', {}).get('roundtrip_us') is not None:
        summary['single_iter_latency_us'] = out['iter0']['roundtrip_us']
    if summary:
        out['summary'] = summary
    return out


class _Tee:
    """File-like fan-out so the console stream goes to both a buffer and a log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _wait_for_shell(child, *, boot_timeout: float, poke_interval: float = 15) -> None:
    r"""Wait for the (pre-logged-in) guest shell to become responsive.

    The image autologins and hands us a root shell, so there's no username or
    password to send -- we just confirm a live shell by bouncing a unique ``echo``
    marker and matching it as standalone shell *output* (anchored by newlines,
    tolerating the multi-CR ``\r\r\n`` line endings this serial console emits),
    never the echoed input line. That avoids treating the tty's echo of our own
    keystrokes during boot as a ready shell -- which would fire the host command
    into the still-booting console and wedge the run.
    """
    marker = f'AIE4ML_RDY_{os.getpid()}'
    ready = r'\r*\n' + re.escape(marker) + r'\r*\n'

    # Wait until we're past the bootloader (kernel/userspace) before nudging the
    # console, so we can't accidentally interrupt U-Boot autoboot.
    child.expect(_BOOTED_RE, timeout=boot_timeout)

    # Probe until a real shell echoes our marker back on its own line.
    deadline = time.monotonic() + boot_timeout
    child.sendline(f'echo {marker}')
    while time.monotonic() < deadline:
        idx = child.expect([ready, pexpect.TIMEOUT], timeout=poke_interval)
        if idx == 0:                         # shell echoed our marker -> ready
            return
        child.sendline(f'echo {marker}')     # TIMEOUT -> nudge again

    raise pexpect.TIMEOUT(f'no responsive shell within {boot_timeout}s')


def _shutdown(child, timeout: float) -> None:
    """Power off the guest and make sure the QEMU process tree is reaped."""
    try:
        if child.isalive():
            child.sendline('poweroff')
            child.expect(pexpect.EOF, timeout=timeout)
    except Exception:
        pass
    finally:
        pid = getattr(child, 'pid', None)
        try:
            child.close(force=True)
        except Exception:
            pass
        # launch_emulator forks QEMU children; the pexpect child is a session
        # leader, so killing its process group cleans up any orphans.
        if pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass


def run_hw_emu(
    output_dir,
    *,
    n_iter: int = 1,
    xclbin: str = 'a.xclbin',
    sd_mount=('/run/media/mmcblk0p1', '/mnt'),
    extra_launch_args: str = '-add-env AIE_COMPILER_WORKDIR=../Work',
    boot_timeout: float = 1800,
    run_timeout: float = 1800,
    shutdown_timeout: float = 120,
    log_path=None,
    echo: bool = False,
) -> Dict[str, Any]:
    """Boot the hw_emu QEMU image, run the host, and return its parsed perf.

    Parameters mirror the manual flow: it ``cd``s into ``<output_dir>/output``,
    runs ``launch_hw_emu.sh``, waits for the autologin shell, executes
    ``./host.exe <xclbin> <n_iter>`` from ``sd_mount``, waits for the host's
    completion sentinel, then powers the guest off.

    Returns a nested dict ``{'host': {...}, 'pl': {...}, 'iter0': {...},
    'summary': {'iteration_sample_latency_us', 'single_iter_latency_us'},
    'n_iter': int, 'log_path': str}``. Raises if the package is missing or no
    performance output could be parsed (full console log is at ``log_path``).

    Pass ``echo=True`` to mirror the live QEMU console to stdout while it runs --
    the quickest way to see where boot/login stalls. On timeout/EOF the last
    console output is also included in the raised error.
    """
    if pexpect is None:
        raise RuntimeError(
            'pexpect is required for hardware-emulation runs; install it with '
            '`pip install pexpect`.'
        ) from _PEXPECT_IMPORT_ERROR

    run_dir = Path(output_dir) / 'output'
    launch = run_dir / _LAUNCH_SCRIPT
    if not launch.exists():
        raise FileNotFoundError(
            f'{launch} not found. Build the hw_emu package first: '
            'aie_model.build(make_target="hw_emu").'
        )

    n_iter = int(n_iter)
    if log_path is None:
        log_path = run_dir / 'hw_emu_run.log'

    # Mirror `make run_emu`: `cd ./output && ./launch_hw_emu.sh -add-env ...`.
    # Run it through `bash -c` (as make does) rather than spawning the script by
    # a relative path -- pexpect resolves a './' command against the *process*
    # CWD, not spawn(cwd=...), so a bare './launch_hw_emu.sh' is reported as
    # "not found or was not executable".
    launch_cmd = (
        f'cd {shlex.quote(str(run_dir))} && ./{_LAUNCH_SCRIPT} {extra_launch_args}'.strip()
    )
    log.info('Launching hw_emu: %s', launch_cmd)

    transcript = io.StringIO()
    child = pexpect.spawn(
        '/bin/bash',
        ['-c', launch_cmd],
        encoding='utf-8',
        codec_errors='replace',
        timeout=run_timeout,
        dimensions=(50, 200),
    )
    log_file = open(log_path, 'w')
    streams = [transcript, log_file]
    if echo:
        # Mirror the live console to stdout so a stalled boot/login is visible.
        streams.append(sys.stdout)
    child.logfile_read = _Tee(*streams)

    try:
        # Wait for the pre-logged-in (autologin) shell to become responsive.
        _wait_for_shell(child, boot_timeout=boot_timeout)

        # Set the emulation env and cd into whichever candidate mount actually has
        # host.exe (the SD card auto-mounts at /run/media/mmcblk0p1, but some boots
        # land at /mnt), then run it. No sudo/mount needed.
        mounts = [sd_mount] if isinstance(sd_mount, str) else list(sd_mount)
        candidates = ' '.join(shlex.quote(m) for m in mounts)
        host_cmd = (
            'export XILINX_XRT=/usr XCL_EMULATION_MODE=hw_emu; '
            f'for d in {candidates}; do [ -x "$d/{_HOST_EXE}" ] && cd "$d" && break; done; '
            f'./{_HOST_EXE} {xclbin} {n_iter}'
        )
        log.info('hw_emu guest command: %s', host_cmd)
        child.sendline(host_cmd)
        child.expect(_DONE_RE, timeout=run_timeout)
    except (pexpect.TIMEOUT, pexpect.EOF) as exc:
        kind = (
            'timed out (no expected prompt/sentinel matched)'
            if isinstance(exc, pexpect.TIMEOUT)
            else 'emulator exited before the run completed'
        )
        tail = transcript.getvalue()[-2000:]
        raise RuntimeError(
            f'hw_emu {kind}; full log at {log_path}.\n'
            f'--- last console output ---\n{tail}'
        ) from exc
    finally:
        text = transcript.getvalue()
        _shutdown(child, shutdown_timeout)
        log_file.close()

    perf = parse_perf(text)
    if not perf:
        raise RuntimeError(
            f'hw_emu run produced no parseable performance output; see {log_path}.'
        )
    perf['n_iter'] = n_iter
    perf['log_path'] = str(log_path)
    return perf
