/*
 * spectral_flux.h - Spectral Flux calculation for onset/syllable detection
 *
 * Spectral Flux measures the rate of change of the power spectrum,
 * effective for detecting onsets including unvoiced consonants.
 */

#ifndef SPECTRAL_FLUX_H
#define SPECTRAL_FLUX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
typedef struct SpectralFlux SpectralFlux;

/*
 * Initialize Spectral Flux calculator
 *
 * @param sample_rate   Audio sample rate (Hz)
 * @param fft_size      FFT window size in samples (must be power of 2)
 * @param hop_size      Hop size in samples
 * @param custom_alloc  Custom allocator (NULL for default malloc)
 * @return              Initialized SpectralFlux object or NULL on failure
 */
SpectralFlux *spectral_flux_create(int sample_rate, int fft_size, int hop_size,
                                   void *(*custom_alloc)(size_t));

/*
 * Process audio samples and compute spectral flux
 *
 * @param sf            SpectralFlux object
 * @param input         Input audio samples (mono, float)
 * @param num_samples   Number of input samples
 * @param flux_out      Output buffer for flux values (one per hop)
 * @param max_flux      Maximum number of flux values to output
 * @return              Number of flux values written
 */
int spectral_flux_process(SpectralFlux *sf, const float *input, int num_samples,
                          float *flux_out, int max_flux);

/*
 * Get the current instantaneous spectral flux value
 * (useful for sample-by-sample processing)
 */
float spectral_flux_get_current(const SpectralFlux *sf);

/*
 * Reset internal state
 */
void spectral_flux_reset(SpectralFlux *sf);

/*
 * Destroy and free resources
 */
void spectral_flux_destroy(SpectralFlux *sf, void (*custom_free)(void *));

/*
 * Get current Spectral Flatness (0 = harmonic/vowel, 1 = noise/consonant)
 */
float spectral_flux_get_flatness(const SpectralFlux *sf);

/*
 * Get Weber ratio of flatness change
 * Negative = becoming more harmonic (vowel onset)
 * Positive = becoming more noisy (consonant)
 */
float spectral_flux_get_flatness_weber(const SpectralFlux *sf);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_FLUX_H */
