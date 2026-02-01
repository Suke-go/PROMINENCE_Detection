#ifndef AGC_H
#define AGC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AgcState AgcState;

// Create AGC instance
// target_db: Target RMS level in dB (e.g., -20.0)
// max_gain_db: Maximum amplification in dB (e.g., 30.0)
AgcState *agc_create(int sample_rate, float target_db, float max_gain_db,
                     void *(*alloc_fn)(size_t));

// Destroy AGC instance
void agc_destroy(AgcState *agc, void (*free_fn)(void *));

// Reset AGC state
void agc_reset(AgcState *agc);

// Process a single sample
// Returns the gain-adjusted sample
float agc_process(AgcState *agc, float sample);

// Get current gain (linear)
float agc_get_gain(AgcState *agc);

#ifdef __cplusplus
}
#endif

#endif // AGC_H
