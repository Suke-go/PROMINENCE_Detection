/*
 * spectral_flux.c - Spectral Flux implementation with SIMD optimization
 *
 * Computes the half-wave rectified spectral flux:
 *   SF[n] = sum( max(0, |X[n,k]| - |X[n-1,k]|)^2 )
 *
 * This captures onset transients, including unvoiced consonants.
 */

#include "spectral_flux.h"
#include "../../extern/kissfft/kiss_fft.h"
#include "../../extern/kissfft/kiss_fftr.h"
#include "simd_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct SpectralFlux {
  int sample_rate;
  int fft_size;
  int hop_size;
  int n_bins; /* fft_size/2 + 1 */

  /* FFT state */
  kiss_fftr_cfg fft_cfg;

  /* Buffers */
  float *input_buffer; /* Ring buffer for input samples */
  int input_write_pos;
  int samples_since_hop;

  float *window;          /* Hann window */
  float *windowed_frame;  /* Windowed frame for FFT */
  kiss_fft_cpx *spectrum; /* Current spectrum */
  float *prev_magnitude;  /* Previous frame magnitude */
  float *curr_magnitude;  /* Current frame magnitude */

  /* Output */
  float current_flux;
  float current_flatness; /* Spectral Flatness (0=harmonic, 1=noise) */
  float prev_flatness;    /* Previous flatness (for Weber ratio) */
  float flatness_weber;   /* Weber ratio of flatness change */

  /* Memory */
  void *(*alloc_fn)(size_t);
};

/* Generate Hann window */
static void generate_hann_window(float *window, int size) {
  for (int i = 0; i < size; i++) {
    window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (size - 1)));
  }
}

SpectralFlux *spectral_flux_create(int sample_rate, int fft_size, int hop_size,
                                   void *(*custom_alloc)(size_t)) {
  void *(*alloc)(size_t) = custom_alloc ? custom_alloc : malloc;

  SpectralFlux *sf = (SpectralFlux *)alloc(sizeof(SpectralFlux));
  if (!sf)
    return NULL;

  memset(sf, 0, sizeof(SpectralFlux));
  sf->sample_rate = sample_rate;
  sf->fft_size = fft_size;
  sf->hop_size = hop_size;
  sf->n_bins = fft_size / 2 + 1;
  sf->alloc_fn = alloc;

  /* Initialize FFT */
  sf->fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
  if (!sf->fft_cfg) {
    alloc == malloc ? free(sf) : (void)0;
    return NULL;
  }

  /* Allocate buffers */
  sf->input_buffer = (float *)alloc(fft_size * sizeof(float));
  sf->window = (float *)alloc(fft_size * sizeof(float));
  sf->windowed_frame = (float *)alloc(fft_size * sizeof(float));
  sf->spectrum = (kiss_fft_cpx *)alloc(sf->n_bins * sizeof(kiss_fft_cpx));
  sf->prev_magnitude = (float *)alloc(sf->n_bins * sizeof(float));
  sf->curr_magnitude = (float *)alloc(sf->n_bins * sizeof(float));

  if (!sf->input_buffer || !sf->window || !sf->windowed_frame ||
      !sf->spectrum || !sf->prev_magnitude || !sf->curr_magnitude) {
    spectral_flux_destroy(sf, alloc == malloc ? free : NULL);
    return NULL;
  }

  /* Initialize */
  memset(sf->input_buffer, 0, fft_size * sizeof(float));
  memset(sf->prev_magnitude, 0, sf->n_bins * sizeof(float));
  memset(sf->curr_magnitude, 0, sf->n_bins * sizeof(float));
  generate_hann_window(sf->window, fft_size);

  sf->input_write_pos = 0;
  sf->samples_since_hop = 0;
  sf->current_flux = 0.0f;

  return sf;
}

void spectral_flux_reset(SpectralFlux *sf) {
  if (!sf)
    return;

  memset(sf->input_buffer, 0, sf->fft_size * sizeof(float));
  memset(sf->prev_magnitude, 0, sf->n_bins * sizeof(float));
  memset(sf->curr_magnitude, 0, sf->n_bins * sizeof(float));
  sf->input_write_pos = 0;
  sf->samples_since_hop = 0;
  sf->current_flux = 0.0f;
}

void spectral_flux_destroy(SpectralFlux *sf, void (*custom_free)(void *)) {
  if (!sf)
    return;

  void (*free_fn)(void *) = custom_free ? custom_free : free;

  if (sf->fft_cfg)
    kiss_fftr_free(sf->fft_cfg);
  if (sf->input_buffer)
    free_fn(sf->input_buffer);
  if (sf->window)
    free_fn(sf->window);
  if (sf->windowed_frame)
    free_fn(sf->windowed_frame);
  if (sf->spectrum)
    free_fn(sf->spectrum);
  if (sf->prev_magnitude)
    free_fn(sf->prev_magnitude);
  if (sf->curr_magnitude)
    free_fn(sf->curr_magnitude);

  free_fn(sf);
}

/* Compute spectral flux for current frame */
static float compute_flux(SpectralFlux *sf) {
  /* Apply window (SIMD optimized) */
  memcpy(sf->windowed_frame, sf->input_buffer, sf->fft_size * sizeof(float));

  /* Rearrange ring buffer to linear frame */
  /* Actually, we need to read fft_size samples ending at write_pos */
  int read_start = sf->input_write_pos; /* This is where oldest sample is */
  for (int i = 0; i < sf->fft_size; i++) {
    int idx = (read_start + i) % sf->fft_size;
    sf->windowed_frame[i] = sf->input_buffer[idx];
  }

  simd_apply_window_f32(sf->windowed_frame, sf->window, sf->fft_size);

  /* FFT */
  kiss_fftr(sf->fft_cfg, sf->windowed_frame, sf->spectrum);

  /* Compute magnitude and Spectral Flatness simultaneously */
  float log_sum = 0.0f;   /* For geometric mean (sum of logs) */
  float arith_sum = 0.0f; /* For arithmetic mean */
  int valid_bins = 0;

  for (int k = 1; k < sf->n_bins; k++) { /* Skip DC bin */
    float r = sf->spectrum[k].r;
    float i = sf->spectrum[k].i;
    float mag = sqrtf(r * r + i * i);
    sf->curr_magnitude[k] = mag;

    if (mag > 1e-10f) {
      log_sum += logf(mag);
      arith_sum += mag;
      valid_bins++;
    }
  }
  sf->curr_magnitude[0] = 0.0f; /* Zero DC */

  /* Spectral Flatness = exp(mean(log(mag))) / mean(mag)
   * = geometric_mean / arithmetic_mean
   * Range: 0 (pure tone) to 1 (white noise) */
  float flatness = 0.0f;
  if (valid_bins > 0 && arith_sum > 1e-10f) {
    float geom_mean = expf(log_sum / valid_bins);
    float arith_mean = arith_sum / valid_bins;
    flatness = geom_mean / arith_mean;
    if (flatness > 1.0f)
      flatness = 1.0f;
  }

  /* Weber ratio for flatness change (perceptual saliency)
   * Negative change = becoming more harmonic = likely vowel onset */
  float flatness_change = flatness - sf->prev_flatness;
  float weber_denom = sf->prev_flatness + 0.01f; /* Avoid division by zero */
  sf->flatness_weber = flatness_change / weber_denom;

  sf->prev_flatness = flatness;
  sf->current_flatness = flatness;

  /* Half-wave rectified spectral flux (SIMD optimized) */
  float flux =
      simd_hwr_diff_sum_f32(sf->curr_magnitude, sf->prev_magnitude, sf->n_bins);

  /* Normalize by number of bins */
  flux /= sf->n_bins;

  /* Swap magnitude buffers */
  float *tmp = sf->prev_magnitude;
  sf->prev_magnitude = sf->curr_magnitude;
  sf->curr_magnitude = tmp;

  return flux;
}

int spectral_flux_process(SpectralFlux *sf, const float *input, int num_samples,
                          float *flux_out, int max_flux) {
  int flux_count = 0;

  for (int i = 0; i < num_samples; i++) {
    /* Add sample to ring buffer */
    sf->input_buffer[sf->input_write_pos] = input[i];
    sf->input_write_pos = (sf->input_write_pos + 1) % sf->fft_size;
    sf->samples_since_hop++;

    /* Compute flux every hop_size samples */
    if (sf->samples_since_hop >= sf->hop_size) {
      sf->samples_since_hop = 0;
      sf->current_flux = compute_flux(sf);

      if (flux_out && flux_count < max_flux) {
        flux_out[flux_count++] = sf->current_flux;
      }
    }
  }

  return flux_count;
}

float spectral_flux_get_current(const SpectralFlux *sf) {
  return sf ? sf->current_flux : 0.0f;
}

float spectral_flux_get_flatness(const SpectralFlux *sf) {
  return sf ? sf->current_flatness : 0.0f;
}

float spectral_flux_get_flatness_weber(const SpectralFlux *sf) {
  return sf ? sf->flatness_weber : 0.0f;
}
