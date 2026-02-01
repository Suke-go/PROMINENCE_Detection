#include "agc.h"
#include <math.h>
#include <stdlib.h>

struct AgcState {
  float target_level; // Linear target RMS
  float max_gain;     // Linear max gain
  float current_gain;

  // Envelope follower state
  float envelope;
  float attack_coeff;  // For rising signal (finding peak) -> Fast
  float release_coeff; // For falling signal -> Slow

  // Gain smoothing
  float gain_coeff;

  void *(*alloc_fn)(size_t);
  void (*free_fn)(void *);
};

static void *default_malloc(size_t size) { return malloc(size); }
static void default_free(void *ptr) { free(ptr); }

AgcState *agc_create(int sample_rate, float target_db, float max_gain_db,
                     void *(*alloc_fn)(size_t)) {
  if (!alloc_fn)
    alloc_fn = default_malloc;

  AgcState *agc = (AgcState *)alloc_fn(sizeof(AgcState));
  if (!agc)
    return NULL;

  agc->alloc_fn = alloc_fn;
  agc->free_fn = default_free;

  // Convert dB to linear
  agc->target_level = powf(10.0f, target_db / 20.0f);
  agc->max_gain = powf(10.0f, max_gain_db / 20.0f);

  // Initialize state
  agc->current_gain = 1.0f;
  agc->envelope = 0.0f;

  // Time constants
  // Envelope: Fast attack (5ms), Slower release (500ms)
  float t_att = 0.005f;
  float t_rel = 0.500f;
  agc->attack_coeff = 1.0f - expf(-1.0f / (t_att * sample_rate));
  agc->release_coeff = 1.0f - expf(-1.0f / (t_rel * sample_rate));

  // Gain smoothing (100ms) to prevent zipper noise
  agc->gain_coeff = 1.0f - expf(-1.0f / (0.100f * sample_rate));

  return agc;
}

void agc_destroy(AgcState *agc, void (*free_fn)(void *)) {
  if (!agc)
    return;
  if (free_fn)
    free_fn(agc);
  else
    free(agc);
}

void agc_reset(AgcState *agc) {
  agc->current_gain = 1.0f;
  agc->envelope = 0.0f;
}

float agc_process(AgcState *agc, float sample) {
  // 1. Estimate signal envelope (rectified)
  float abs_sample = fabsf(sample);

  if (abs_sample > agc->envelope) {
    agc->envelope += agc->attack_coeff * (abs_sample - agc->envelope);
  } else {
    agc->envelope += agc->release_coeff * (abs_sample - agc->envelope);
  }

  // 2. Calculate target gain
  // Avoid division by zero
  float env_safe = (agc->envelope > 1e-6f) ? agc->envelope : 1e-6f;
  float target_gain = agc->target_level / env_safe;

  // 3. Limit gain
  if (target_gain > agc->max_gain)
    target_gain = agc->max_gain;
  if (target_gain < 0.1f)
    target_gain = 0.1f; // Don't attenuate too much

  // 4. Smooth gain update
  agc->current_gain += agc->gain_coeff * (target_gain - agc->current_gain);

  // 5. Apply gain
  return sample * agc->current_gain;
}

float agc_get_gain(AgcState *agc) { return agc->current_gain; }
