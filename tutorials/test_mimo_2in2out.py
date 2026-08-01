import hls4ml
import numpy as np
import tensorflow as tf
from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# ===========================================================================================
# 2-input / 2-output variant of tutorial_3.py (same flow; only build_model() changes).
#
# Two INDEPENDENT branches: a -> y1 and b -> y2. The inputs never combine, so no merge layer
# is needed -- which is the ONLY way a multi-INPUT graph lowers through the Keras frontend
# (a merge op such as Add/Concatenate raises "No family resolver registered for op_type='merge'").
#
# In memory_stream mode this emits ONE mm2s per input tensor AND one s2mm per output tensor,
# each sized to its own tensor:
#     a (256) -> mm2s   ;  b (128) -> mm2s_2      (different-sized input movers)
#     y1(128) -> s2mm   ;  y2(64)  -> s2mm_2      (different-sized output movers)
#
# For inputs that must INTERACT (e.g. Add(a, b)), the Keras frontend can't express it -- use the
# ONNX frontend instead: tutorials/test_mimo_onnx.py.
# ===========================================================================================

np.random.seed(42)
tf.random.set_seed(42)

N_A = 256   # input a features
N_B = 128   # input b features
BATCH = 256
ITERS = 4
PLATFORM = 'xilinx_vek280_base_202520_1'

PROJECT_NAME = 'mimo2_hardware_new'


def build_model():
    a = tf.keras.Input(batch_size=BATCH, shape=(N_A,), name='a')
    b = tf.keras.Input(batch_size=BATCH, shape=(N_B,), name='b')

    # branch A: a -> dense(128) -> y1
    xa = QActivation(quantized_bits(8, 0), name='quant_a')(a)
    xa = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_a')(xa)
    xa = QActivation(quantized_bits(8, 0), name='quant_a2')(xa)
    out_a = QActivation(quantized_bits(8, 0), name='quant_out_a')(xa)

    # branch B: b -> dense(64) -> y2  (independent of branch A -> no merge op)
    xb = QActivation(quantized_bits(8, 0), name='quant_b')(b)
    xb = QDense(64, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_b')(xb)
    xb = QActivation(quantized_bits(8, 0), name='quant_b2')(xb)
    out_b = QActivation(quantized_bits(8, 0), name='quant_out_b')(xb)

    return Model([a, b], [out_a, out_b], name='mimo_2in2out')


model = build_model()
model.compile(optimizer=Adam(1e-3), loss='mse')
model.summary()

# ── HLS config ──
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Match tutorial_3's result-precision override, applied to both dense branches when present.
for dense in ('dense_a', 'dense_b'):
    lin = f'{dense}_linear'
    if lin in cfg.get('LayerName', {}):
        cfg['LayerName'][lin].setdefault('Precision', {})['result'] = 'fixed<8,3,TRN,WRAP,0>'

print('\nLayer precision summary:')
for name, layer_cfg in cfg.get('LayerName', {}).items():
    print(f"  {name}: {layer_cfg.get('Precision', {})}")

# ── AIE conversion ──
# target='hardware' enables PL data mover + XRT host emission (default 'aie' = AIE-only).
aie_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir='proj_aie_' + PROJECT_NAME,
    backend='aie',
    project_name='proj_aie_' + PROJECT_NAME,
    batch_size=BATCH,
    iterations=ITERS,
    part=PLATFORM,
    target='hardware',  # hardware | aie
    pl_memory='uram',  # uram | bram
    enable_pl_timing=True,  # True | False (auto-disabled with >1 mover on either side)
    pl_data_mover_mode='memory_stream',  # benchmark | memory_stream | external_stream
)

aie_model.compile()
# By default the simulation works for the hardware target. To build for hardware emulation,
# use aie_model.build(make_target='hw_emu')
aie_model.build()

# Simulation -- multi-input X is a dict keyed by input tensor name; multi-output result is a dict.
x = {
    'a': np.random.random((BATCH, N_A)).astype(np.float32),
    'b': np.random.random((BATCH, N_B)).astype(np.float32),
}
y_aie = aie_model.predict(x, simulator='aie')

print('\n' + '=' * 60)
print('2-INPUT / 2-OUTPUT SIMULATION')
print('=' * 60)
outputs = y_aie if isinstance(y_aie, dict) else {'out': y_aie}
for name, y in outputs.items():
    print(f'  output {name}: shape {np.asarray(y).shape}')

report = read_aie_report(aie_model)

print('\n' + '=' * 60)
print('AIE SIMULATION REPORT')
print('=' * 60)

if 'throughput' in report:
    t = report['throughput']
    print('\n[Throughput]')
    print(f"  Avg : {t['Avg_GOPs']} GOPs")
    print(f"  Min : {t['Min_GOPs']} GOPs")
    print(f"  Max : {t['Max_GOPs']} GOPs")

if 'output_interval' in report:
    ii = report['output_interval']
    print('\n[Output Interval (ns)]')
    for name, vals in ii.items():
        if isinstance(vals, dict):
            print(f'  {name}:')
            for k, v in vals.items():
                print(f'    {k}: {v} ns')
