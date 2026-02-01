/*
 * simd_utils.h - SIMD utility functions for libsyllable
 *
 * Provides cross-platform SIMD abstractions with fallback to scalar code.
 * Supports SSE2/SSE4, AVX2, and NEON (ARM).
 */

#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <math.h>
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

/* --- Platform Detection --- */

#if defined(__SSE2__) || defined(_M_X64) ||                                    \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SIMD_SSE2 1
#include <emmintrin.h>
#endif

#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
#define SIMD_SSE4 1
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#define SIMD_AVX2 1
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_NEON 1
#include <arm_neon.h>
#endif

/* --- Vector Operations --- */

/*
 * simd_dot_product_f32 - Compute dot product of two float arrays
 * SIMD optimized with scalar fallback
 */
static inline float simd_dot_product_f32(const float *a, const float *b,
                                         size_t n) {
  float sum = 0.0f;
  size_t i = 0;

#if defined(SIMD_AVX2)
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    vsum = _mm256_fmadd_ps(va, vb, vsum);
  }
  /* Horizontal sum */
  __m128 hi = _mm256_extractf128_ps(vsum, 1);
  __m128 lo = _mm256_castps256_ps128(vsum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum = _mm_cvtss_f32(sum128);

#elif defined(SIMD_SSE2)
  __m128 vsum = _mm_setzero_ps();
  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
  }
  /* Horizontal sum */
  __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
  vsum = _mm_add_ps(vsum, shuf);
  shuf = _mm_movehl_ps(shuf, vsum);
  vsum = _mm_add_ss(vsum, shuf);
  sum = _mm_cvtss_f32(vsum);

#elif defined(SIMD_NEON)
  float32x4_t vsum = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vsum = vmlaq_f32(vsum, va, vb);
  }
  /* Horizontal sum */
  float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
  vsum2 = vpadd_f32(vsum2, vsum2);
  sum = vget_lane_f32(vsum2, 0);
#endif

  /* Scalar tail */
  for (; i < n; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

/*
 * simd_sum_squares_f32 - Compute sum of squares (L2 norm squared)
 */
static inline float simd_sum_squares_f32(const float *a, size_t n) {
  float sum = 0.0f;
  size_t i = 0;

#if defined(SIMD_AVX2)
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    vsum = _mm256_fmadd_ps(va, va, vsum);
  }
  __m128 hi = _mm256_extractf128_ps(vsum, 1);
  __m128 lo = _mm256_castps256_ps128(vsum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum = _mm_cvtss_f32(sum128);

#elif defined(SIMD_SSE2)
  __m128 vsum = _mm_setzero_ps();
  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(va, va));
  }
  __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
  vsum = _mm_add_ps(vsum, shuf);
  shuf = _mm_movehl_ps(shuf, vsum);
  vsum = _mm_add_ss(vsum, shuf);
  sum = _mm_cvtss_f32(vsum);

#elif defined(SIMD_NEON)
  float32x4_t vsum = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    vsum = vmlaq_f32(vsum, va, va);
  }
  float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
  vsum2 = vpadd_f32(vsum2, vsum2);
  sum = vget_lane_f32(vsum2, 0);
#endif

  for (; i < n; i++) {
    sum += a[i] * a[i];
  }

  return sum;
}

/*
 * simd_hwr_diff_sum_f32 - Half-wave rectified difference sum (for Spectral
 * Flux) Computes: sum(max(0, a[i] - b[i])^2)
 */
static inline float simd_hwr_diff_sum_f32(const float *a, const float *b,
                                          size_t n) {
  float sum = 0.0f;
  size_t i = 0;

#if defined(SIMD_SSE2)
  __m128 vsum = _mm_setzero_ps();
  __m128 vzero = _mm_setzero_ps();
  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 diff = _mm_sub_ps(va, vb);
    diff = _mm_max_ps(diff, vzero); /* Half-wave rectify */
    vsum = _mm_add_ps(vsum, _mm_mul_ps(diff, diff));
  }
  __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
  vsum = _mm_add_ps(vsum, shuf);
  shuf = _mm_movehl_ps(shuf, vsum);
  vsum = _mm_add_ss(vsum, shuf);
  sum = _mm_cvtss_f32(vsum);

#elif defined(SIMD_NEON)
  float32x4_t vsum = vdupq_n_f32(0.0f);
  float32x4_t vzero = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    diff = vmaxq_f32(diff, vzero);
    vsum = vmlaq_f32(vsum, diff, diff);
  }
  float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
  vsum2 = vpadd_f32(vsum2, vsum2);
  sum = vget_lane_f32(vsum2, 0);
#endif

  for (; i < n; i++) {
    float diff = a[i] - b[i];
    if (diff > 0.0f) {
      sum += diff * diff;
    }
  }

  return sum;
}

/*
 * simd_apply_window_f32 - Apply window function to signal (in-place)
 */
static inline void simd_apply_window_f32(float *data, const float *window,
                                         size_t n) {
  size_t i = 0;

#if defined(SIMD_AVX2)
  for (; i + 8 <= n; i += 8) {
    __m256 vd = _mm256_loadu_ps(data + i);
    __m256 vw = _mm256_loadu_ps(window + i);
    _mm256_storeu_ps(data + i, _mm256_mul_ps(vd, vw));
  }
#elif defined(SIMD_SSE2)
  for (; i + 4 <= n; i += 4) {
    __m128 vd = _mm_loadu_ps(data + i);
    __m128 vw = _mm_loadu_ps(window + i);
    _mm_storeu_ps(data + i, _mm_mul_ps(vd, vw));
  }
#elif defined(SIMD_NEON)
  for (; i + 4 <= n; i += 4) {
    float32x4_t vd = vld1q_f32(data + i);
    float32x4_t vw = vld1q_f32(window + i);
    vst1q_f32(data + i, vmulq_f32(vd, vw));
  }
#endif

  for (; i < n; i++) {
    data[i] *= window[i];
  }
}

/*
 * simd_magnitude_f32 - Compute magnitude of complex array
 * Input: cpx[n] as interleaved [r0,i0,r1,i1,...]
 * Output: mag[n/2]
 */
static inline void simd_magnitude_f32(const float *cpx, float *mag,
                                      size_t n_complex) {
  for (size_t i = 0; i < n_complex; i++) {
    float r = cpx[2 * i];
    float im = cpx[2 * i + 1];
    mag[i] = sqrtf(r * r + im * im);
  }
}

#ifdef __cplusplus
}
#endif

#endif /* SIMD_UTILS_H */
