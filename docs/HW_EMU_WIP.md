
## Adding hardware emulation flow to the AIE4ML pipeline

Hardware emulation is wired through Python via `predict(simulator='hw_emu')`,
which boots QEMU, waits for the autologin shell, runs the host, captures the
performance it prints, and powers off — returning the report as a dict (no OFM
read-back, so the input arg is ignored):

```python
perf = aie_model.predict(x, simulator='hw_emu')   # after build(make_target='hw_emu')
print(perf['summary']['iteration_sample_latency_us'],
      perf['summary']['single_iter_latency_us'])
```

It can still be driven manually from the CLI if preferred:
```bash
make run_emu
# After the system boots in QEMU, run
./host.exe a.xclbin <ITERATIONS>
```

## Current WIP and issues
- Basic setup for passing hw_emu flow as a python input is wired
- pexpect does not always work and cannot read qemu's petalinux terminal status
- Still finalizing on how to make the hw_emu run automatically execute the design and return perf numbers