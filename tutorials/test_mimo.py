import hls4ml
import numpy as np
import tensorflow as tf
from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# ===========================================================================================
# Multi-OUTPUT variant of tutorial_3.py (same flow; the only change is build_model()).
#
# One input fans out into TWO independent heads -> two graph OUTPUT tensors. In memory_stream
# mode this emits one s2mm output mover PER output tensor (s2mm + s2mm_2), each sized to its
# own tensor -- the output-side mirror of the multi-input mm2s movers.
#
# NOTE: multi-INPUT is NOT possible from Keras. A 2-input Keras model has to combine its inputs
# with a merge layer (Add/Concatenate), which the hls4ml->aie frontend cannot lower
# ("No family resolver registered for op_type='merge'"). For a true multi-INPUT + multi-OUTPUT
# design, use tutorials/test_mimo_onnx.py (ONNX frontend, which does lower Add).
# ===========================================================================================

np.random.seed(42)
tf.random.set_seed(42)

N_IN = 256
BATCH = 256
ITERS = 4
PLATFORM = 'xilinx_vek280_base_202520_1'

PROJECT_NAME = 'mimo_hardware_new'


def build_model():
    inp = tf.keras.Input(batch_size=BATCH, shape=(N_IN,), name='inp')
    x = QActivation(quantized_bits(8, 0), name='input_quant')(inp)
    # --- MULTI-OUTPUT: two heads fan out from the shared input (different widths -> different-
    #     sized s2mm movers). No merge op, so this lowers through the Keras frontend. ---
    a = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_a')(x)
    a = QActivation(quantized_bits(8, 0), name='quant_a')(a)
    out_a = QActivation(quantized_bits(8, 0), name='quant_out_a')(a)

    b = QDense(64, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_b')(x)
    b = QActivation(quantized_bits(8, 0), name='quant_b')(b)
    out_b = QActivation(quantized_bits(8, 0), name='quant_out_b')(b)

    return Model(inp, [out_a, out_b], name='multi_output_demo')


model = build_model()
model.compile(optimizer=Adam(1e-3), loss='mse')
model.summary()

# ── HLS config ──
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Match tutorial_3's result-precision override, applied to both dense heads when present.
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

# Simulation
x = np.random.random((BATCH, N_IN)).astype(np.float32)
y_aie = aie_model.predict(x, simulator='aie')  # list of arrays (one per output tensor)

print('\n' + '=' * 60)
print('MULTI-OUTPUT SIMULATION')
print('=' * 60)
outputs = y_aie if isinstance(y_aie, (list, tuple)) else [y_aie]
for i, y in enumerate(outputs):
    print(f'  output {i}: shape {np.asarray(y).shape}')

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
