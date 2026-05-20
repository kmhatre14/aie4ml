// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "softmax.h"

using namespace adf;

template <typename ConfigT>
softmax_i8<ConfigT>::softmax_i8(
    int16_t B_i,
    int8_t S_i,
    uint8_t DMAX_i)
    : softmax_base<ConfigT>(),
      B_param(B_i),
      S_param(S_i),
      DMAX_param(DMAX_i)
{}

template <typename ConfigT>
inline __attribute__((always_inline)) void softmax_i8<ConfigT>::softmax_row(
    const int8* __restrict in_ptr,
    out_t* __restrict out_ptr,
    int16_t B,
    int8_t S,
    uint8_t DMAX) {
    constexpr int VEC = ConfigT::VEC;
    constexpr int VECS = COLS / VEC;

    auto vin_it = aie::cbegin_vector<VEC>(in_ptr);
    aie::vector<int8, VEC> vmax = *vin_it++;
    for (int i = 1; i < VECS; ++i) {
        vmax = aie::max(vmax, *vin_it++);
    }
    int8 max_val = aie::reduce_max(vmax);

    vin_it = aie::cbegin_vector<VEC>(in_ptr);

    auto scratch_it = aie::begin_vector<VEC>(scratch);
    aie::accum<acc32, VEC> acc_sum = aie::zeros<acc32, VEC>();

    uint8 max_u = (uint8)max_val;
    aie::vector<uint8, VEC> max_u_vec = aie::broadcast<uint8, VEC>(max_u);

    for (int i = 0; i < VECS; ++i) {
        aie::vector<int8, VEC> x = *vin_it++;
        aie::vector<uint8, VEC> xu = x.template cast_to<uint8>();
        aie::vector<uint8, VEC> d = aie::sub(max_u_vec, xu);
        d = aie::min(d, (uint8)DMAX);

        aie::vector<int8, VEC> d8 = d.template cast_to<int8>();
        aie::accum<acc32, VEC> acc;
        acc.from_vector(aie::broadcast<int32, VEC>(B));
        acc = aie::mac(acc, d8, aie::broadcast<int8, VEC>(-S));

        acc_sum = aie::add(acc_sum, acc);
        aie::vector<int16, VEC> score16 = acc.template to_vector<int16>(0);
        *scratch_it++ = score16;
    }

    int32_t sum = aie::reduce_add(acc_sum.template to_vector<int32_t>());

    int32_t inv_q0;
    if constexpr (ConfigT::USE_CLB) {
        int leading_zeros = clb(sum);
        int k = 31 - leading_zeros;
        if constexpr (std::is_same_v<out_t, int16_t>) {
            inv_q0 = 32767 >> k;
        } else {
            inv_q0 = 255 << (INV_SHIFT - k);
        }
    } else {
        if constexpr (std::is_same_v<out_t, int16_t>) {
            inv_q0 = 32767 / sum;
        } else {
            inv_q0 = (255 << INV_SHIFT) / sum;
        }
    }

    aie::vector<int16, VEC> inv_vec = aie::broadcast<int16, VEC>(inv_q0);

    auto scratch_rd = aie::cbegin_vector<VEC>(scratch);
    auto out_it = aie::begin_vector<VEC>(out_ptr);
    for (int i = 0; i < VECS; ++i) {
        aie::vector<int16, VEC> v = *scratch_rd++;
        aie::accum<acc32, VEC> prod = aie::mul(v, inv_vec);
        *out_it++ = prod.template to_vector<out_t>(OUT_SHIFT);
    }
}

template <typename ConfigT>
void softmax_i8<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out) {
    auto in_ptr = (const in_t*)in.data();
    auto out_ptr = (out_t*)out.data();

    const int16_t B = B_param;
    const int8_t S = S_param;
    const uint8_t DMAX = DMAX_param;
    for (int row = 0; row < ROWS; ++row) {
        softmax_row(in_ptr + row * COLS, out_ptr + row * COLS, B, S, DMAX);
    }
}
