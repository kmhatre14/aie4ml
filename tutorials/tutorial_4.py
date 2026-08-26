import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from qkeras import QDense, QActivation, quantized_bits

import hls4ml
from keras.utils import plot_model

np.random.seed(42)
tf.random.set_seed(42)

N_IN  = 256
BATCH = 256
ITERS = 4
PLATFORM = 'xilinx_vek280_base_202520_1'

PROJECT_NAME = 'model_2d_1s_2d_1s_1d_sm_pl'

N_DENSE = 5            # dense_0, dense_1, dense_2
SOFTMAX_COLS = 128     # features the softmax reduces over (== dense_1 output)

# HCCS Softmax calibration constants (integer surrogate, not exponential softmax).
# The AIE kernel scores each element as  clamp(B - S * (max - x), min=0)  with the
# distance saturated at Dmax, then normalizes. Constraints enforced by the resolver
# (see op_impls/families/softmax/common.py::pack_hccs_params) for cols=SOFTMAX_COLS:
#   B in [0,32767], S in [0,127], Dmax in [0,127]
#   B - S*Dmax >= 0                      -> 255 - 2*100 = 55   OK
#   cols * (B - S*Dmax) >= 256           -> 128 * 55 = 7040    OK  (reciprocal range)
#   cols * B <= 32767                    -> 128 * 255 = 32640  OK  (sum fits int16)
# Retune these if you change SOFTMAX_COLS.
HCCS = {'B': 255, 'S': 2, 'Dmax': 100}


def build_model():
    """2 dense layers -> softmax -> 1 dense layer.

    256 -> 128 -> 128 -> [softmax over 128] -> 64
    The softmax axis (128) must be a multiple of the int8 vector size (32).
    """
    inp = tf.keras.Input(batch_size=BATCH, shape=(N_IN,), name='inp')
    x = QActivation(quantized_bits(8, 0), name='input_quant')(inp)

    # ── 2 dense layers ──
    x = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_0')(x)
    x = QActivation(quantized_bits(8, 0), name='quant_0')(x)
    x = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_1')(x)
    x = QActivation(quantized_bits(8, 0), name='quant_1')(x)   # int8 -> softmax input

    # ── softmax ──
    x = tf.keras.layers.Activation('softmax', name='softmax_0')(x)

    # ── final dense layer ──
    x = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_2')(x)
    x = QActivation(quantized_bits(8, 0), name='quant_2')(x)
    x = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_3')(x)
    x = QActivation(quantized_bits(8, 0), name='quant_3')(x)   # int8 -> softmax input    

    # ── softmax ──
    x = tf.keras.layers.Activation('softmax', name='softmax_1')(x)

    # ── final dense layer ──
    x = QDense(64, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_4')(x)
    out = QActivation(quantized_bits(8, 0), name='quant_out')(x)
    return Model(inp, out, name='dense2_softmax_dense')

model = build_model()
model.compile(optimizer=Adam(1e-3), loss='mse')
model.summary()

model.save(PROJECT_NAME + ".h5")
# ── HLS config ──
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')

for h in range(N_DENSE):
    cfg['LayerName'][f'dense_{h}_linear']['Precision']['result'] = 'fixed<8,3,TRN,WRAP,0>'

# HCCS Softmax: needs the approximation family + its calibration constants, and the
# resolver requires a signed int8 input (quant_1) and a uint8 or int16 output.
cfg['LayerName']['softmax_0']['approximation'] = 'hccs'
# cfg['LayerName']['softmax_0']['hccs'] = dict(HCCS)
cfg['LayerName']['softmax_0'].setdefault('Precision', {})['result'] = 'ufixed<8,0,TRN,WRAP,0>'
cfg['LayerName']['softmax_0'].update({'approximation': 'hccs', 'hccs': dict(HCCS)})

cfg['LayerName']['softmax_1']['approximation'] = 'hccs'
# cfg['LayerName']['softmax_1']['hccs'] = dict(HCCS)
cfg['LayerName']['softmax_1'].setdefault('Precision', {})['result'] = 'ufixed<8,0,TRN,WRAP,0>'
cfg['LayerName']['softmax_1'].update({'approximation': 'hccs', 'hccs': dict(HCCS)})

# Pin BOTH ends of each direct dense->dense pair so a spacer column stays between them
# (dense_0: cols 7-10, dense_1: 12-13 -> col 11 free; dense_2: 12-13, dense_3: 15-16 -> col 14 free).
# Pinning only the consumer is not enough: the placer slides the free producer up against it.
cfg['LayerName']['dense_0']['placement'] = {'col': 7, 'row': 0}
cfg['LayerName']['dense_1']['placement'] = {'col': 12, 'row': 0}
cfg['LayerName']['dense_2']['placement'] = {'col': 12, 'row': 4}
cfg['LayerName']['dense_3']['placement'] = {'col': 15, 'row': 4}

print('\nLayer precision summary:')
for name, layer_cfg in cfg.get('LayerName', {}).items():
    print(f"  {name}: {layer_cfg.get('Precision', {})}")

# Make the softmax a PL kernel
cfg['LayerName']['softmax_0']['run_on'] = 'pl'
cfg['LayerName']['softmax_1']['run_on'] = 'pl'

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
    target='hardware',          # hardware | aie
    pl_memory='bram',            # uram | bram
    enable_pl_timing = True,    # True | False 
    pl_data_mover_mode = 'memory_stream' # benchmark | memory_stream | external_stream
)

# aie_model.compile()
# By default the simulation works for hardware target, To build for hardware_emulation use aie_model.build(make_target='hw_emu')
aie_model.build()

# Simulation
# x = np.random.random((BATCH, N_IN)).astype(np.float32)
# y_aie = aie_model.predict(x, simulator='aie')[:BATCH]

# from aie4ml.simulation import read_aie_report
# report = read_aie_report(aie_model)

# print('\n' + '=' * 60)
# print('AIE SIMULATION REPORT')
# print('=' * 60)

# if 'throughput' in report:
#     t = report['throughput']
#     print('\n[Throughput]')
#     print(f"  Avg : {t['Avg_GOPs']} GOPs")
#     print(f"  Min : {t['Min_GOPs']} GOPs")
#     print(f"  Max : {t['Max_GOPs']} GOPs")

# if 'output_interval' in report:
#     ii = report['output_interval']
#     print('\n[Output Interval (ns)]')
#     for name, vals in ii.items():
#         if isinstance(vals, dict):
#             print(f"  {name}:")
#             for k, v in vals.items():
#                 print(f"    {k}: {v} ns")
