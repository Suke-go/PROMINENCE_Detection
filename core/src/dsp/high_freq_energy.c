/*
 * high_freq_energy.c - High-frequency energy tracker implementation
 *
 * Uses a 2nd order Butterworth high-pass filter followed by
 * exponential smoothing for energy estimation.
 *
 * Target frequency range: 2-8kHz (fricatives, plosive bursts)
 */

#include "high_freq_energy.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct HighFreqEnergy {
  int sample_rate;
  float cutoff_hz;

  /* 2nd order Butterworth high-pass filter state */
  float b0, b1, b2; /* Numerator coefficients */
  float a1, a2;     /* Denominator coefficients (normalized) */
  float x1, x2;     /* Input history */
  float y1, y2;     /* Output history */

  /* Energy smoothing */
  float energy;
  float attack_coef;
  float release_coef;

  /* Peak tracking for transients */
  float peak_energy;
  float peak_decay;
};

/*
 * Design 2nd order Butterworth high-pass filter coefficients
 */
static void design_highpass(HighFreqEnergy *hfe) {
  float fc = hfe->cutoff_hz;
  float fs = (float)hfe->sample_rate;

  /* Pre-warp the cutoff frequency */
  float wc = tanf((float)M_PI * fc / fs);
  float wc2 = wc * wc;

  /* Butterworth Q = 1/sqrt(2) for 2nd order */
  float sqrt2 = 1.41421356237f;

  /* Bilinear transform coefficients for high-pass */
  float k = 1.0f + sqrt2 * wc + wc2;

  hfe->b0 = 1.0f / k;
  hfe->b1 = -2.0f / k;
  hfe->b2 = 1.0f / k;
  hfe->a1 = 2.0f * (wc2 - 1.0f) / k;
  hfe->a2 = (1.0f - sqrt2 * wc + wc2) / k;
}

HighFreqEnergy *hfe_create(int sample_rate, float cutoff_hz, float window_ms,
                           void *(*custom_alloc)(size_t)) {
  void *(*alloc)(size_t) = custom_alloc ? custom_alloc : malloc;

  HighFreqEnergy *hfe = (HighFreqEnergy *)alloc(sizeof(HighFreqEnergy));
  if (!hfe)
    return NULL;

  memset(hfe, 0, sizeof(HighFreqEnergy));
  hfe->sample_rate = sample_rate;
  hfe->cutoff_hz = cutoff_hz > 0 ? cutoff_hz : 2000.0f;

  /* Design filter */
  design_highpass(hfe);

  /* Energy smoothing coefficients
   * Attack: fast (1ms) to catch transients
   * Release: slower (window_ms) for smoothing
   */
  float attack_ms = 1.0f;
  float release_ms = window_ms > 0 ? window_ms : 10.0f;

  hfe->attack_coef = 1.0f - expf(-1.0f / (sample_rate * attack_ms * 0.001f));
  hfe->release_coef = 1.0f - expf(-1.0f / (sample_rate * release_ms * 0.001f));

  /* Peak decay: ~50ms */
  hfe->peak_decay = 1.0f - expf(-1.0f / (sample_rate * 0.05f));

  return hfe;
}

void hfe_reset(HighFreqEnergy *hfe) {
  if (!hfe)
    return;

  hfe->x1 = hfe->x2 = 0.0f;
  hfe->y1 = hfe->y2 = 0.0f;
  hfe->energy = 0.0f;
  hfe->peak_energy = 0.0f;
}

void hfe_destroy(HighFreqEnergy *hfe, void (*custom_free)(void *)) {
  if (!hfe)
    return;
  void (*free_fn)(void *) = custom_free ? custom_free : free;
  free_fn(hfe);
}

float hfe_process(HighFreqEnergy *hfe, float input) {
  /* Apply high-pass filter (Direct Form II Transposed) */
  float filtered = hfe->b0 * input + hfe->b1 * hfe->x1 + hfe->b2 * hfe->x2 -
                   hfe->a1 * hfe->y1 - hfe->a2 * hfe->y2;

  /* Update history */
  hfe->x2 = hfe->x1;
  hfe->x1 = input;
  hfe->y2 = hfe->y1;
  hfe->y1 = filtered;

  /* Compute instantaneous energy */
  float inst_energy = filtered * filtered;

  /* Attack/release envelope follower */
  if (inst_energy > hfe->energy) {
    hfe->energy += hfe->attack_coef * (inst_energy - hfe->energy);
  } else {
    hfe->energy += hfe->release_coef * (inst_energy - hfe->energy);
  }

  /* Track peak for transient detection */
  if (hfe->energy > hfe->peak_energy) {
    hfe->peak_energy = hfe->energy;
  } else {
    hfe->peak_energy -= hfe->peak_decay * hfe->peak_energy;
  }

  return hfe->energy;
}

float hfe_get_current(const HighFreqEnergy *hfe) {
  return hfe ? hfe->energy : 0.0f;
}
