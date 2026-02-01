#ifndef WAVELET_H
#define WAVELET_H

#include "syllable_detector.h" // For config or types if needed
#include <stddef.h>
#include <stdint.h>


// Wavelet Module based on Morlet Wavelet
// Used for robust transient/onset detection particularly for unvoiced
// consonants

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WaveletDetector WaveletDetector;

// Create a new Wavelet Detector
// sample_rate: Audio sample rate
// min_freq: Minimum frequency to check (e.g., 2000Hz for unvoiced)
// max_freq: Maximum frequency to check (e.g., 8000Hz)
// num_scales: Number of scales (frequencies) to analyze
WaveletDetector *wavelet_create(int sample_rate, float min_freq, float max_freq,
                                int num_scales, void *(*alloc_fn)(size_t));

// Destroy the detector
void wavelet_destroy(WaveletDetector *wd, void (*free_fn)(void *));

// Reset state
void wavelet_reset(WaveletDetector *wd);

// Process audio samples
// Returns the "transient score" (0.0 to 1.0+) indicating likelihood of an onset
float wavelet_process(WaveletDetector *wd, float sample);

// Get current energy at a specific scale index
float wavelet_get_energy(WaveletDetector *wd, int scale_idx);

#ifdef __cplusplus
}
#endif

#endif // WAVELET_H
