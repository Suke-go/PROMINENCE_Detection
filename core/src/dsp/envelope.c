#include "envelope.h"
#include <math.h>

void envelope_init(EnvelopeFollower *e, float sample_rate, float attack_ms,
                   float release_ms) {
  e->output = 0.0f;

  // Time constant formula: coeff = exp(-1.0 / (time_ms * 0.001 * sample_rate))
  // Or simpler approximation for high sample rates: 1.0 / (time_samples)
  // We'll use the standard exponential decay formula.

  float t_attack = attack_ms * 0.001f;
  float t_release = release_ms * 0.001f;

  // Ensure non-zero to avoid division by zero
  if (t_attack < 1.0e-5f)
    t_attack = 1.0e-5f;
  if (t_release < 1.0e-5f)
    t_release = 1.0e-5f;

  e->attack_coeff = expf(-1.0f / (sample_rate * t_attack));
  e->release_coeff = expf(-1.0f / (sample_rate * t_release));
}

float envelope_process(EnvelopeFollower *e, float in) {
  float abs_in = fabsf(in);

  if (abs_in > e->output) {
    e->output = e->attack_coeff * e->output + (1.0f - e->attack_coeff) * abs_in;
  } else {
    e->output =
        e->release_coeff * e->output + (1.0f - e->release_coeff) * abs_in;
  }

  return e->output;
}
