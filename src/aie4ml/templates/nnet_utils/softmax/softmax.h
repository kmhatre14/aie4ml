// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
#include <type_traits>

#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class softmax_base {
public:
    softmax_base() {
#if defined(__AIENGINE__) && (__cplusplus >= 202002L)
        aie::set_rounding(ConfigT::ROUNDING);
        aie::set_saturation(ConfigT::SATURATION);
#endif
    }
};

// HCCS is a calibrated clipped-linear surrogate for attention softmax.
// It is intentionally integer-only. see https://arxiv.org/pdf/2604.02292v1
template <typename ConfigT>
class softmax_i8 : public softmax_base<ConfigT> {
public:
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS      = ConfigT::ROWS;
    static constexpr int COLS      = ConfigT::COLS;
    static constexpr int VEC       = ConfigT::VEC;
    static constexpr int VECS      = COLS / VEC;
    static constexpr int INV_SHIFT = ConfigT::INV_SHIFT;
    static constexpr int OUT_SHIFT = std::is_same_v<out_t, int16_t> ? 0 : INV_SHIFT;

    softmax_i8(int16_t B_i, int8_t S_i, uint8_t DMAX_i);

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_i8::run);
    }

private:
    alignas(aie::vector_decl_align) int16_t scratch[COLS];

    int16_t B_param;
    int8_t S_param;
    uint8_t DMAX_param;

    inline __attribute__((always_inline))
    void softmax_row(const int8* __restrict in_ptr,
                     out_t* __restrict out_ptr,
                     int16_t B,
                     int8_t S,
                     uint8_t DMAX);
};
