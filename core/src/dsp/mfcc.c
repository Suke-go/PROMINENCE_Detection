/*
 * mfcc.c - MFCC implementation optimized for onset/syllable detection
 *
 * Pipeline:
 *   1. Windowing (Hann)
 *   2. FFT → Power spectrum
 *   3. Mel filterbank → Log energy
 *   4. DCT → MFCC coefficients
 *   5. Delta computation → L2 norm
 *
 * The delta-MFCC magnitude is particularly useful for detecting
 * phoneme transitions and syllable onsets.
 */

#include "mfcc.h"
#include "../../extern/kissfft/kiss_fft.h"
#include "../../extern/kissfft/kiss_fftr.h"
#include "simd_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct MFCC {
  int sample_rate;
  int fft_size;
  int hop_size;
  int n_bins;

  /* FFT */
  kiss_fftr_cfg fft_cfg;

  /* Buffers */
  float *input_buffer;
  int input_write_pos;
  int samples_since_hop;

  float *window;
  float *windowed_frame;
  kiss_fft_cpx *spectrum;
  float *power_spectrum;

  /* Mel filterbank */
  float *mel_energies;
  float **mel_filters;   /* [MFCC_NUM_FILTERS][n_bins] */
  int *mel_filter_start; /* Start bin for each filter */
  int *mel_filter_end;   /* End bin for each filter */

  /* DCT matrix for MFCC */
  float *dct_matrix; /* [MFCC_NUM_COEFFS][MFCC_NUM_FILTERS] */

  /* Output */
  float coeffs[MFCC_NUM_COEFFS];
  float prev_coeffs[MFCC_NUM_COEFFS];
  float delta_magnitude;

  /* Memory */
  void *(*alloc_fn)(size_t);
};

/* Convert frequency to Mel scale */
static float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/* Convert Mel scale to frequency */
static float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* Initialize Mel filterbank */
static int init_mel_filterbank(MFCC *m) {
  float mel_low = hz_to_mel(80.0f);                         /* 80 Hz low edge */
  float mel_high = hz_to_mel((float)m->sample_rate / 2.0f); /* Nyquist */

  /* Mel points equally spaced */
  float mel_points[MFCC_NUM_FILTERS + 2];
  for (int i = 0; i < MFCC_NUM_FILTERS + 2; i++) {
    mel_points[i] = mel_low + (mel_high - mel_low) * i / (MFCC_NUM_FILTERS + 1);
  }

  /* Convert to Hz and then to FFT bin indices */
  int hz_points[MFCC_NUM_FILTERS + 2];
  float bin_width = (float)m->sample_rate / m->fft_size;
  for (int i = 0; i < MFCC_NUM_FILTERS + 2; i++) {
    float hz = mel_to_hz(mel_points[i]);
    hz_points[i] = (int)(hz / bin_width + 0.5f);
    if (hz_points[i] >= m->n_bins)
      hz_points[i] = m->n_bins - 1;
  }

  /* Create triangular filters */
  for (int f = 0; f < MFCC_NUM_FILTERS; f++) {
    int start = hz_points[f];
    int center = hz_points[f + 1];
    int end = hz_points[f + 2];

    m->mel_filter_start[f] = start;
    m->mel_filter_end[f] = end;

    for (int k = 0; k < m->n_bins; k++) {
      if (k < start || k > end) {
        m->mel_filters[f][k] = 0.0f;
      } else if (k <= center) {
        /* Rising slope */
        m->mel_filters[f][k] = (float)(k - start) / (center - start + 1);
      } else {
        /* Falling slope */
        m->mel_filters[f][k] = (float)(end - k) / (end - center + 1);
      }
    }
  }

  return 0;
}

/* Initialize DCT matrix */
static void init_dct_matrix(MFCC *m) {
  /* Type-II DCT, orthonormal */
  float scale = sqrtf(2.0f / MFCC_NUM_FILTERS);
  for (int i = 0; i < MFCC_NUM_COEFFS; i++) {
    for (int j = 0; j < MFCC_NUM_FILTERS; j++) {
      m->dct_matrix[i * MFCC_NUM_FILTERS + j] =
          scale * cosf((float)M_PI * i * (j + 0.5f) / MFCC_NUM_FILTERS);
    }
  }
}

MFCC *mfcc_create(int sample_rate, int fft_size, int hop_size,
                  void *(*custom_alloc)(size_t)) {
  void *(*alloc)(size_t) = custom_alloc ? custom_alloc : malloc;

  MFCC *m = (MFCC *)alloc(sizeof(MFCC));
  if (!m)
    return NULL;

  memset(m, 0, sizeof(MFCC));
  m->sample_rate = sample_rate;
  m->fft_size = fft_size;
  m->hop_size = hop_size;
  m->n_bins = fft_size / 2 + 1;
  m->alloc_fn = alloc;

  /* FFT */
  m->fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
  if (!m->fft_cfg)
    goto fail;

  /* Buffers */
  m->input_buffer = (float *)alloc(fft_size * sizeof(float));
  m->window = (float *)alloc(fft_size * sizeof(float));
  m->windowed_frame = (float *)alloc(fft_size * sizeof(float));
  m->spectrum = (kiss_fft_cpx *)alloc(m->n_bins * sizeof(kiss_fft_cpx));
  m->power_spectrum = (float *)alloc(m->n_bins * sizeof(float));
  m->mel_energies = (float *)alloc(MFCC_NUM_FILTERS * sizeof(float));
  m->dct_matrix =
      (float *)alloc(MFCC_NUM_COEFFS * MFCC_NUM_FILTERS * sizeof(float));

  /* Mel filterbank */
  m->mel_filters = (float **)alloc(MFCC_NUM_FILTERS * sizeof(float *));
  m->mel_filter_start = (int *)alloc(MFCC_NUM_FILTERS * sizeof(int));
  m->mel_filter_end = (int *)alloc(MFCC_NUM_FILTERS * sizeof(int));

  if (!m->input_buffer || !m->window || !m->windowed_frame || !m->spectrum ||
      !m->power_spectrum || !m->mel_energies || !m->dct_matrix ||
      !m->mel_filters || !m->mel_filter_start || !m->mel_filter_end) {
    goto fail;
  }

  for (int f = 0; f < MFCC_NUM_FILTERS; f++) {
    m->mel_filters[f] = (float *)alloc(m->n_bins * sizeof(float));
    if (!m->mel_filters[f])
      goto fail;
  }

  /* Initialize */
  memset(m->input_buffer, 0, fft_size * sizeof(float));
  memset(m->coeffs, 0, sizeof(m->coeffs));
  memset(m->prev_coeffs, 0, sizeof(m->prev_coeffs));

  /* Hann window */
  for (int i = 0; i < fft_size; i++) {
    m->window[i] =
        0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fft_size - 1)));
  }

  init_mel_filterbank(m);
  init_dct_matrix(m);

  return m;

fail:
  mfcc_destroy(m, alloc == malloc ? free : NULL);
  return NULL;
}

void mfcc_reset(MFCC *m) {
  if (!m)
    return;

  memset(m->input_buffer, 0, m->fft_size * sizeof(float));
  memset(m->coeffs, 0, sizeof(m->coeffs));
  memset(m->prev_coeffs, 0, sizeof(m->prev_coeffs));
  m->input_write_pos = 0;
  m->samples_since_hop = 0;
  m->delta_magnitude = 0.0f;
}

void mfcc_destroy(MFCC *m, void (*custom_free)(void *)) {
  if (!m)
    return;

  void (*free_fn)(void *) = custom_free ? custom_free : free;

  if (m->fft_cfg)
    kiss_fftr_free(m->fft_cfg);
  if (m->input_buffer)
    free_fn(m->input_buffer);
  if (m->window)
    free_fn(m->window);
  if (m->windowed_frame)
    free_fn(m->windowed_frame);
  if (m->spectrum)
    free_fn(m->spectrum);
  if (m->power_spectrum)
    free_fn(m->power_spectrum);
  if (m->mel_energies)
    free_fn(m->mel_energies);
  if (m->dct_matrix)
    free_fn(m->dct_matrix);

  if (m->mel_filters) {
    for (int f = 0; f < MFCC_NUM_FILTERS; f++) {
      if (m->mel_filters[f])
        free_fn(m->mel_filters[f]);
    }
    free_fn(m->mel_filters);
  }
  if (m->mel_filter_start)
    free_fn(m->mel_filter_start);
  if (m->mel_filter_end)
    free_fn(m->mel_filter_end);

  free_fn(m);
}

/* Compute MFCC for current frame */
static void compute_mfcc(MFCC *m) {
  /* Rearrange ring buffer */
  int read_start = m->input_write_pos;
  for (int i = 0; i < m->fft_size; i++) {
    int idx = (read_start + i) % m->fft_size;
    m->windowed_frame[i] = m->input_buffer[idx];
  }

  /* Apply window */
  simd_apply_window_f32(m->windowed_frame, m->window, m->fft_size);

  /* FFT */
  kiss_fftr(m->fft_cfg, m->windowed_frame, m->spectrum);

  /* Power spectrum */
  for (int k = 0; k < m->n_bins; k++) {
    float r = m->spectrum[k].r;
    float im = m->spectrum[k].i;
    m->power_spectrum[k] = r * r + im * im;
  }

  /* Apply Mel filterbank */
  for (int f = 0; f < MFCC_NUM_FILTERS; f++) {
    float energy = 0.0f;
    int start = m->mel_filter_start[f];
    int end = m->mel_filter_end[f];

    for (int k = start; k <= end; k++) {
      energy += m->power_spectrum[k] * m->mel_filters[f][k];
    }

    /* Log compression (add small epsilon for numerical stability) */
    m->mel_energies[f] = logf(energy + 1e-10f);
  }

  /* Save previous coefficients for delta */
  memcpy(m->prev_coeffs, m->coeffs, sizeof(m->coeffs));

  /* DCT to get MFCC (SIMD optimized dot products) */
  for (int i = 0; i < MFCC_NUM_COEFFS; i++) {
    m->coeffs[i] = simd_dot_product_f32(&m->dct_matrix[i * MFCC_NUM_FILTERS],
                                        m->mel_energies, MFCC_NUM_FILTERS);
  }

  /* Compute delta magnitude (L2 norm of difference) */
  float delta_sum = 0.0f;
  for (int i = 0; i < MFCC_NUM_COEFFS; i++) {
    float d = m->coeffs[i] - m->prev_coeffs[i];
    delta_sum += d * d;
  }
  m->delta_magnitude = sqrtf(delta_sum);
}

int mfcc_process(MFCC *m, const float *input, int num_samples, float *delta_out,
                 int max_delta) {
  int delta_count = 0;

  for (int i = 0; i < num_samples; i++) {
    m->input_buffer[m->input_write_pos] = input[i];
    m->input_write_pos = (m->input_write_pos + 1) % m->fft_size;
    m->samples_since_hop++;

    if (m->samples_since_hop >= m->hop_size) {
      m->samples_since_hop = 0;
      compute_mfcc(m);

      if (delta_out && delta_count < max_delta) {
        delta_out[delta_count++] = m->delta_magnitude;
      }
    }
  }

  return delta_count;
}

void mfcc_get_coeffs(const MFCC *m, float *coeffs_out) {
  if (m && coeffs_out) {
    memcpy(coeffs_out, m->coeffs, MFCC_NUM_COEFFS * sizeof(float));
  }
}

float mfcc_get_delta_magnitude(const MFCC *m) {
  return m ? m->delta_magnitude : 0.0f;
}
