#include "biquad.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void biquad_reset(Biquad *f) {
  f->x1 = f->x2 = f->y1 = f->y2 = 0.0f;
  f->b0 = f->b1 = f->b2 = f->a1 = f->a2 = 0.0f;
}

void biquad_config_bandpass(Biquad *f, float sample_rate, float center_freq,
                            float q_factor) {
  float w0 = 2.0f * (float)M_PI * center_freq / sample_rate;
  float alpha = sinf(w0) / (2.0f * q_factor);

  // Bandpass with constant skirt gain (peak gain = Q)
  // To normalize peak gain to 0dB, we might want a different formulation,
  // but the standard RBJ cookbook for BPF usually is:

  float b0 = alpha;
  float b1 = 0.0f;
  float b2 = -alpha;
  float a0 = 1.0f + alpha;
  float a1 = -2.0f * cosf(w0);
  float a2 = 1.0f - alpha;

  // Normalize by a0
  float inv_a0 = 1.0f / a0;

  f->b0 = b0 * inv_a0;
  f->b1 = b1 * inv_a0;
  f->b2 = b2 * inv_a0;
  f->a1 = a1 * inv_a0;
  f->a2 = a2 * inv_a0;

  // Reset state on reconfig? Usually safer to keep state if just updating
  // coeffs, but for initial setup it doesn't matter.
}

float biquad_process(Biquad *f, float in) {
  // Direct Form I
  float out = f->b0 * in + f->b1 * f->x1 + f->b2 * f->x2 - f->a1 * f->y1 -
              f->a2 * f->y2;

  // Avoid denormals
  if (fabsf(out) < 1.0e-15f)
    out = 0.0f;

  f->x2 = f->x1;
  f->x1 = in;
  f->y2 = f->y1;
  f->y1 = out;

  return out;
}
