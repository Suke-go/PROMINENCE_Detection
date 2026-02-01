/*
 * mfcc.h - Mel-Frequency Cepstral Coefficients for phonetic analysis
 *
 * Provides MFCC calculation and delta-MFCC for detecting
 * spectral shape changes at phoneme boundaries.
 */

#ifndef MFCC_H
#define MFCC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MFCC MFCC;

/* Number of MFCC coefficients (excluding C0) */
#define MFCC_NUM_COEFFS 13

/* Number of Mel filter banks */
#define MFCC_NUM_FILTERS 26

/*
 * Create MFCC calculator
 *
 * @param sample_rate   Audio sample rate (Hz)
 * @param fft_size      FFT size (must match spectral_flux if sharing)
 * @param hop_size      Hop size in samples
 * @param custom_alloc  Custom allocator (NULL for malloc)
 */
MFCC *mfcc_create(int sample_rate, int fft_size, int hop_size,
                  void *(*custom_alloc)(size_t));

/*
 * Process samples and compute delta-MFCC magnitude
 *
 * @param mfcc          MFCC object
 * @param input         Input audio samples
 * @param num_samples   Number of samples
 * @param delta_out     Output buffer for delta-MFCC L2 norms (one per hop)
 * @param max_delta     Maximum output values
 * @return              Number of values written
 */
int mfcc_process(MFCC *mfcc, const float *input, int num_samples,
                 float *delta_out, int max_delta);

/*
 * Get current MFCC coefficients
 *
 * @param mfcc          MFCC object
 * @param coeffs_out    Output buffer (must be at least MFCC_NUM_COEFFS)
 */
void mfcc_get_coeffs(const MFCC *mfcc, float *coeffs_out);

/*
 * Get current delta-MFCC magnitude (L2 norm of coefficient changes)
 */
float mfcc_get_delta_magnitude(const MFCC *mfcc);

/*
 * Reset internal state
 */
void mfcc_reset(MFCC *mfcc);

/*
 * Destroy and free resources
 */
void mfcc_destroy(MFCC *mfcc, void (*custom_free)(void *));

#ifdef __cplusplus
}
#endif

#endif /* MFCC_H */
