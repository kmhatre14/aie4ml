#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class layernorm_base {
public:
    layernorm_base() {
#if defined(__AIENGINE__) && (__cplusplus >= 202002L)
        aie::set_rounding(ConfigT::ROUNDING);
        aie::set_saturation(ConfigT::SATURATION);
#endif
    }
};

// layernorm_i8 — fully-integer LayerNorm, int8 → int8.
template <typename ConfigT>
class layernorm_i8 : public layernorm_base<ConfigT> {
public:
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS        = ConfigT::ROWS;
    static constexpr int COLS        = ConfigT::COLS;
    static constexpr int VEC         = ConfigT::VEC;
    static constexpr int VECS        = COLS / VEC;
    static constexpr int GAMMA_SHIFT = ConfigT::GAMMA_SHIFT;
    static constexpr int OUT_SHIFT   = ConfigT::OUT_SHIFT;
    static constexpr int NORM_SHIFT  = 15;

    static constexpr int LOG2_COLS = []() constexpr {
        int n = COLS, c = 0;
        while (n > 1) { n >>= 1; ++c; }
        return c;
    }();

    layernorm_i8();

    void run(input_buffer<in_t>&           in,
             const int16_t (&gamma)[COLS],
             const int16_t (&beta)[COLS],
             output_buffer<out_t>&         out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(layernorm_i8::run);
    }

private:
    alignas(aie::vector_decl_align)
    static constexpr uint16_t invsqrt_seed_even_lut[64] = {
        32641, 32391, 32146, 31907, 31673, 31445, 31221, 31002,
        30787, 30577, 30371, 30169, 29972, 29778, 29587, 29401,
        29217, 29038, 28861, 28688, 28518, 28350, 28186, 28024,
        27866, 27709, 27556, 27405, 27256, 27110, 26966, 26825,
        26686, 26548, 26413, 26280, 26149, 26020, 25893, 25767,
        25644, 25522, 25402, 25283, 25167, 25051, 24938, 24826,
        24715, 24606, 24498, 24392, 24287, 24184, 24081, 23980,
        23881, 23782, 23685, 23589, 23494, 23400, 23307, 23216
    };

    alignas(aie::vector_decl_align)
    static constexpr uint16_t invsqrt_seed_odd_lut[64] = {
        23080, 22904, 22731, 22562, 22396, 22235, 22077, 21922,
        21770, 21621, 21476, 21333, 21193, 21056, 20921, 20789,
        20660, 20533, 20408, 20285, 20165, 20047, 19930, 19816,
        19704, 19594, 19485, 19378, 19273, 19170, 19068, 18968,
        18870, 18773, 18677, 18583, 18490, 18399, 18309, 18220,
        18133, 18047, 17962, 17878, 17795, 17714, 17634, 17554,
        17476, 17399, 17323, 17248, 17174, 17100, 17028, 16957,
        16886, 16817, 16748, 16680, 16613, 16546, 16481, 16416
    };

    static inline __attribute__((always_inline)) int32_t isqrt_q15(int32_t var);

    inline __attribute__((always_inline))
    void layernorm_row(const in_t*    __restrict in_ptr,
                             out_t*   __restrict out_ptr,
                       const int16_t* __restrict gamma_ptr,
                       const int16_t* __restrict beta_ptr);
};
