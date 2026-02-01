/*
 * high_freq_energy.h - High-frequency energy tracker for consonant detection
 *
 * Tracks energy in the 2-8kHz band where fricatives and plosive bursts
 * have significant energy content.
 */

#ifndef HIGH_FREQ_ENERGY_H
#define HIGH_FREQ_ENERGY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HighFreqEnergy HighFreqEnergy;

/*
 * Create high-frequency energy tracker
 *
 * @param sample_rate   Audio sample rate (Hz)
 * @param cutoff_hz     High-pass filter cutoff (default: 2000)
 * @param window_ms     Energy integration window in ms (default: 10)
 * @param custom_alloc  Custom allocator (NULL for malloc)
 */
HighFreqEnergy *hfe_create(int sample_rate, float cutoff_hz, float window_ms,
                           void *(*custom_alloc)(size_t));

/*
 * Process a single sample, returns smoothed high-freq energy
 */
float hfe_process(HighFreqEnergy *hfe, float input);

/*
 * Get current energy value without processing
 */
float hfe_get_current(const HighFreqEnergy *hfe);

/*
 * Reset internal state
 */
void hfe_reset(HighFreqEnergy *hfe);

/*
 * Destroy and free resources
 */
void hfe_destroy(HighFreqEnergy *hfe, void (*custom_free)(void *));

#ifdef __cplusplus
}
#endif

#endif /* HIGH_FREQ_ENERGY_H */
