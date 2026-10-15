/* dlssnr_simd.h

   Copyright (c) 2003-2026 HandBrake Team
   This file is part of the HandBrake source code
   Homepage: <http://handbrake.fr/>.
   It may be used under the terms of the GNU General Public License v2.
   For full terms see the file COPYING file or visit http://www.gnu.org/licenses/gpl-2.0.html

   The DLSS-NR filter's per-frame pixel loops: the picture into and out of
   libdlssnr's interleaved float RGB, the history's release (the cut measure,
   the change at each pixel, the dilation and the composition's stand-in for
   `previous`). Vectorised the way visualboyadvance-m's dlssnr.cpp vectorises
   the same loops: SSE2 is baseline on x86-64 and NEON on AArch64, so the
   vector loops need no run-time check, and the scalar loop under each one
   finishes the row -- and is all there is elsewhere.

   Every vector loop produces the bytes its scalar loop does: no fused
   multiply-adds, no reciprocals in place of divisions, and a sum of
   differences is carried in doubles as the scalar loop carries it (in
   another order, which the one consumer, a threshold, cannot tell). Only
   NaN, which the scalar comparisons let through, can come out otherwise:
   SSE2's min and max take the second operand, so it clips to 0 there, and
   NEON's pass it through.
 */

#ifndef HANDBRAKE_DLSSNR_SIMD_H
#define HANDBRAKE_DLSSNR_SIMD_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(DLSSNR_SCALAR)
// the scalar loops alone, so a test can run them beside the vector ones
#elif defined(__SSE2__) || (defined(_M_X64) && !defined(_M_ARM64EC)) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define DLSSNR_SSE2 1
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DLSSNR_NEON 1
#include <arm_neon.h>
#endif

/* -------------------------------------------------------------------------
 * Pixels: `save_image` and `read_png` in nr_frame_main.c
 * ---------------------------------------------------------------------- */

static inline float dlssnr_clip01(float v)
{
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

// clip to [0, 1], then `v * 255 + 0.5` to a byte. The multiply and the add
// are two statements so that no compiler fuses them: the vector loops cannot,
// and the bytes must agree.
static inline uint8_t dlssnr_to_byte(float v)
{
    float t = dlssnr_clip01(v) * 255.0f;
    t = t + 0.5f;
    return (uint8_t)t;
}

// `byte / 255`, a division rather than a multiply by the reciprocal, so the
// float the network sees is the one the command gives it
static inline float dlssnr_from_byte(unsigned v)
{
    return (float)(v & 0xffu) / 255.0f;
}

#if defined(DLSSNR_SSE2)

// Four interleaved pixels [r0 g0 b0 r1][g1 b1 r2 g2][b2 r3 g3 b3] as three
// channels [r0 r1 r2 r3], [g0 g1 g2 g3], [b0 b1 b2 b3], in five shuffles.
// _mm_shuffle_ps takes two lanes of its first operand, then two of its
// second; _MM_SHUFFLE names the lanes last first.
static inline void dlssnr_sse2_to_planes(__m128 p0, __m128 p1, __m128 p2,
                                         __m128 *r, __m128 *g, __m128 *b)
{
    const __m128 t = _mm_shuffle_ps(p1, p2, _MM_SHUFFLE(2, 1, 3, 2));   // r2 g2 r3 g3
    const __m128 u = _mm_shuffle_ps(p0, p1, _MM_SHUFFLE(1, 0, 2, 1));   // g0 b0 g1 b1
    *r = _mm_shuffle_ps(p0, t, _MM_SHUFFLE(2, 0, 3, 0));                // r0 r1 r2 r3
    *g = _mm_shuffle_ps(u, t, _MM_SHUFFLE(3, 1, 2, 0));                 // g0 g1 g2 g3
    *b = _mm_shuffle_ps(u, p2, _MM_SHUFFLE(3, 0, 3, 1));                // b0 b1 b2 b3
}

// the inverse: three channels of four pixels as twelve interleaved floats
static inline void dlssnr_sse2_to_pixels(__m128 r, __m128 g, __m128 b,
                                         __m128 *p0, __m128 *p1, __m128 *p2)
{
    const __m128 t = _mm_shuffle_ps(r, g, _MM_SHUFFLE(2, 0, 2, 0));     // r0 r2 g0 g2
    const __m128 u = _mm_shuffle_ps(g, b, _MM_SHUFFLE(3, 1, 3, 1));     // g1 g3 b1 b3
    const __m128 v = _mm_shuffle_ps(b, r, _MM_SHUFFLE(3, 1, 2, 0));     // b0 b2 r1 r3
    *p0 = _mm_shuffle_ps(t, v, _MM_SHUFFLE(2, 0, 2, 0));                // r0 g0 b0 r1
    *p1 = _mm_shuffle_ps(u, t, _MM_SHUFFLE(3, 1, 2, 0));                // g1 b1 r2 g2
    *p2 = _mm_shuffle_ps(v, u, _MM_SHUFFLE(3, 1, 3, 1));                // b2 r3 g3 b3
}

// dlssnr_clip01 a lane at a time; NaN comes out 0
static inline __m128 dlssnr_sse2_clip01(__m128 v)
{
    return _mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
}

#elif defined(DLSSNR_NEON)

// dlssnr_clip01 a lane at a time. FMAX and FMIN pass NaN through, as the
// scalar comparisons do.
static inline float32x4_t dlssnr_neon_clip01(float32x4_t v)
{
    return vminq_f32(vmaxq_f32(v, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
}

#endif

/* -------------------------------------------------------------------------
 * out[i] = max(out[i], in[i]) over n bytes, sixteen at a time where the
 * vector loop is built; the scalar loop is what the compiler was already
 * making of it. The dilation's one operation.
 * ---------------------------------------------------------------------- */

static inline void dlssnr_max_into(uint8_t *out, const uint8_t *in, int n)
{
    int i = 0;
#if defined(DLSSNR_SSE2)
    for (; i + 16 <= n; i += 16)
    {
        const __m128i a = _mm_loadu_si128((const __m128i *)(out + i));
        const __m128i b = _mm_loadu_si128((const __m128i *)(in + i));
        _mm_storeu_si128((__m128i *)(out + i), _mm_max_epu8(a, b));
    }
#elif defined(DLSSNR_NEON)
    for (; i + 16 <= n; i += 16)
    {
        vst1q_u8(out + i, vmaxq_u8(vld1q_u8(out + i), vld1q_u8(in + i)));
    }
#endif
    for (; i < n; i++)
    {
        if (in[i] > out[i])
        {
            out[i] = in[i];
        }
    }
}

/* -------------------------------------------------------------------------
 * The sum of |a[i] - b[i]| over n floats, each difference taken in double as
 * the scalar loop takes it. The cut measure.
 * ---------------------------------------------------------------------- */

static inline double dlssnr_sum_abs_diff(const float *a, const float *b, size_t n)
{
    double sum = 0.0;
    size_t i = 0;
#if defined(DLSSNR_SSE2)
    const __m128d sign = _mm_castsi128_pd(_mm_srli_epi64(_mm_set1_epi32(-1), 1));
    __m128d acc = _mm_setzero_pd();
    for (; i + 4 <= n; i += 4)
    {
        const __m128 av = _mm_loadu_ps(a + i);
        const __m128 bv = _mm_loadu_ps(b + i);
        __m128d d = _mm_sub_pd(_mm_cvtps_pd(av), _mm_cvtps_pd(bv));
        acc = _mm_add_pd(acc, _mm_and_pd(d, sign));
        d = _mm_sub_pd(_mm_cvtps_pd(_mm_movehl_ps(av, av)),
                       _mm_cvtps_pd(_mm_movehl_ps(bv, bv)));
        acc = _mm_add_pd(acc, _mm_and_pd(d, sign));
    }
    double lanes[2];
    _mm_storeu_pd(lanes, acc);
    sum = lanes[0] + lanes[1];
#elif defined(DLSSNR_NEON)
    float64x2_t acc = vdupq_n_f64(0.0);
    for (; i + 4 <= n; i += 4)
    {
        const float32x4_t av = vld1q_f32(a + i);
        const float32x4_t bv = vld1q_f32(b + i);
        acc = vaddq_f64(acc, vabdq_f64(vcvt_f64_f32(vget_low_f32(av)),
                                       vcvt_f64_f32(vget_low_f32(bv))));
        acc = vaddq_f64(acc, vabdq_f64(vcvt_high_f64_f32(av),
                                       vcvt_high_f64_f32(bv)));
    }
    sum = vaddvq_f64(acc);
#endif
    for (; i < n; i++)
    {
        const double d = (double)a[i] - b[i];
        sum += d < 0.0 ? -d : d;
    }
    return sum;
}

/* -------------------------------------------------------------------------
 * moved[i] = dlssnr_to_byte(max over the channels of |c[3i+k] - b[3i+k]|):
 * how far each pixel of interleaved RGB changed, as a byte.
 * ---------------------------------------------------------------------- */

static inline void dlssnr_moved(const float *c, const float *b, uint8_t *moved,
                                size_t pixels)
{
    size_t i = 0;
#if defined(DLSSNR_SSE2)
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 k255     = _mm_set1_ps(255.0f);
    const __m128 half     = _mm_set1_ps(0.5f);
    for (; i + 16 <= pixels; i += 16)
    {
        __m128i q[4];
        for (int g = 0; g < 4; g++)
        {
            const float *cp = c + 3 * (i + 4 * g);
            const float *bp = b + 3 * (i + 4 * g);
            __m128 d0 = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(cp),     _mm_loadu_ps(bp)),     abs_mask);
            __m128 d1 = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(cp + 4), _mm_loadu_ps(bp + 4)), abs_mask);
            __m128 d2 = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(cp + 8), _mm_loadu_ps(bp + 8)), abs_mask);
            __m128 r, gr, bl;
            dlssnr_sse2_to_planes(d0, d1, d2, &r, &gr, &bl);
            __m128 m = dlssnr_sse2_clip01(_mm_max_ps(_mm_max_ps(r, gr), bl));
            q[g] = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(m, k255), half));
        }
        _mm_storeu_si128((__m128i *)(moved + i),
                         _mm_packus_epi16(_mm_packs_epi32(q[0], q[1]),
                                          _mm_packs_epi32(q[2], q[3])));
    }
#elif defined(DLSSNR_NEON)
    const float32x4_t k255 = vdupq_n_f32(255.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    for (; i + 16 <= pixels; i += 16)
    {
        uint16x4_t q[4];
        for (int g = 0; g < 4; g++)
        {
            const float32x4x3_t cv = vld3q_f32(c + 3 * (i + 4 * g));
            const float32x4x3_t bv = vld3q_f32(b + 3 * (i + 4 * g));
            float32x4_t m = vmaxq_f32(vmaxq_f32(vabdq_f32(cv.val[0], bv.val[0]),
                                                vabdq_f32(cv.val[1], bv.val[1])),
                                      vabdq_f32(cv.val[2], bv.val[2]));
            m = dlssnr_neon_clip01(m);
            q[g] = vmovn_u32(vcvtq_u32_f32(vaddq_f32(vmulq_f32(m, k255), half)));
        }
        vst1q_u8(moved + i, vcombine_u8(vmovn_u16(vcombine_u16(q[0], q[1])),
                                        vmovn_u16(vcombine_u16(q[2], q[3]))));
    }
#endif
    for (; i < pixels; i++)
    {
        float m = 0.0f;
        for (int k = 0; k < 3; k++)
        {
            float d = c[3 * i + k] - b[3 * i + k];
            d = d < 0.0f ? -d : d;
            m = d > m ? d : m;
        }
        moved[i] = dlssnr_to_byte(m);
    }
}

/* -------------------------------------------------------------------------
 * gate[3i+k] = c[3i+k] -/+ m, m = dlssnr_from_byte(moved[i]): the colour
 * pushed away from itself by the change nearby, the composition's stand-in
 * for `previous`.
 * ---------------------------------------------------------------------- */

static inline void dlssnr_gate(const float *c, const uint8_t *moved, float *gate,
                               size_t pixels)
{
    size_t i = 0;
#if defined(DLSSNR_SSE2)
    const __m128  k255 = _mm_set1_ps(255.0f);
    const __m128i zero = _mm_setzero_si128();
    for (; i + 4 <= pixels; i += 4)
    {
        uint32_t word;
        memcpy(&word, moved + i, sizeof(word));
        __m128i mi = _mm_cvtsi32_si128((int)word);
        mi = _mm_unpacklo_epi16(_mm_unpacklo_epi8(mi, zero), zero);
        const __m128 m  = _mm_div_ps(_mm_cvtepi32_ps(mi), k255);
        const __m128 ms[3] = {
            _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 0, 0)),   // m0 m0 m0 m1
            _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 2, 1, 1)),   // m1 m1 m2 m2
            _mm_shuffle_ps(m, m, _MM_SHUFFLE(3, 3, 3, 2)),   // m2 m3 m3 m3
        };
        for (int j = 0; j < 3; j++)
        {
            const __m128 cv = _mm_loadu_ps(c + 3 * i + 4 * j);
            const __m128 ge = _mm_cmpge_ps(cv, ms[j]);
            _mm_storeu_ps(gate + 3 * i + 4 * j,
                          _mm_or_ps(_mm_and_ps(ge, _mm_sub_ps(cv, ms[j])),
                                    _mm_andnot_ps(ge, _mm_add_ps(cv, ms[j]))));
        }
    }
#elif defined(DLSSNR_NEON)
    const float32x4_t k255 = vdupq_n_f32(255.0f);
    for (; i + 4 <= pixels; i += 4)
    {
        uint32_t word;
        memcpy(&word, moved + i, sizeof(word));
        const uint32x4_t mi = vmovl_u16(vget_low_u16(
                                  vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(word)))));
        const float32x4_t m = vdivq_f32(vcvtq_f32_u32(mi), k255);
        float32x4x3_t cv = vld3q_f32(c + 3 * i);
        for (int k = 0; k < 3; k++)
        {
            cv.val[k] = vbslq_f32(vcgeq_f32(cv.val[k], m),
                                  vsubq_f32(cv.val[k], m), vaddq_f32(cv.val[k], m));
        }
        vst3q_f32(gate + 3 * i, cv);
    }
#endif
    for (; i < pixels; i++)
    {
        const float m = dlssnr_from_byte(moved[i]);
        for (int k = 0; k < 3; k++)
        {
            const float ci = c[3 * i + k];
            gate[3 * i + k] = ci >= m ? ci - m : ci + m;
        }
    }
}

/* -------------------------------------------------------------------------
 * A row of swscale's planar float RGB, clipped to [0, 1], as interleaved
 * RGB -- and back.
 * ---------------------------------------------------------------------- */

static inline void dlssnr_interleave(const float *r, const float *g, const float *b,
                                     float *out, int w)
{
    int x = 0;
#if defined(DLSSNR_SSE2)
    for (; x + 4 <= w; x += 4)
    {
        __m128 p0, p1, p2;
        dlssnr_sse2_to_pixels(dlssnr_sse2_clip01(_mm_loadu_ps(r + x)),
                              dlssnr_sse2_clip01(_mm_loadu_ps(g + x)),
                              dlssnr_sse2_clip01(_mm_loadu_ps(b + x)), &p0, &p1, &p2);
        _mm_storeu_ps(out + 3 * x,     p0);
        _mm_storeu_ps(out + 3 * x + 4, p1);
        _mm_storeu_ps(out + 3 * x + 8, p2);
    }
#elif defined(DLSSNR_NEON)
    for (; x + 4 <= w; x += 4)
    {
        float32x4x3_t p;
        p.val[0] = dlssnr_neon_clip01(vld1q_f32(r + x));
        p.val[1] = dlssnr_neon_clip01(vld1q_f32(g + x));
        p.val[2] = dlssnr_neon_clip01(vld1q_f32(b + x));
        vst3q_f32(out + 3 * x, p);
    }
#endif
    for (; x < w; x++)
    {
        out[3 * x + 0] = dlssnr_clip01(r[x]);
        out[3 * x + 1] = dlssnr_clip01(g[x]);
        out[3 * x + 2] = dlssnr_clip01(b[x]);
    }
}

static inline void dlssnr_deinterleave(const float *in, float *r, float *g, float *b,
                                       int w)
{
    int x = 0;
#if defined(DLSSNR_SSE2)
    for (; x + 4 <= w; x += 4)
    {
        __m128 rv, gv, bv;
        dlssnr_sse2_to_planes(_mm_loadu_ps(in + 3 * x), _mm_loadu_ps(in + 3 * x + 4),
                              _mm_loadu_ps(in + 3 * x + 8), &rv, &gv, &bv);
        _mm_storeu_ps(r + x, dlssnr_sse2_clip01(rv));
        _mm_storeu_ps(g + x, dlssnr_sse2_clip01(gv));
        _mm_storeu_ps(b + x, dlssnr_sse2_clip01(bv));
    }
#elif defined(DLSSNR_NEON)
    for (; x + 4 <= w; x += 4)
    {
        const float32x4x3_t p = vld3q_f32(in + 3 * x);
        vst1q_f32(r + x, dlssnr_neon_clip01(p.val[0]));
        vst1q_f32(g + x, dlssnr_neon_clip01(p.val[1]));
        vst1q_f32(b + x, dlssnr_neon_clip01(p.val[2]));
    }
#endif
    for (; x < w; x++)
    {
        r[x] = dlssnr_clip01(in[3 * x + 0]);
        g[x] = dlssnr_clip01(in[3 * x + 1]);
        b[x] = dlssnr_clip01(in[3 * x + 2]);
    }
}

#endif // HANDBRAKE_DLSSNR_SIMD_H
