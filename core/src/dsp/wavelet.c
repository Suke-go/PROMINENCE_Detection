#include "wavelet.h"
#include "simd_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_KERNEL_SIZE 128
#define HISTORY_SIZE 256

// Complex number for Morlet wavelet
typedef struct {
  float r;
  float i;
} ComplexFloat;

// A single scale (frequency) analyzer
typedef struct {
  float freq_hz;
  float scale;
  int kernel_size;
  ComplexFloat *kernel; // [kernel_size]

  // History buffer for convolution
  // We only need history up to kernel_size
  float *input_history;
  int history_idx;

  // Current values
  float current_magnitude;
  float current_energy;
  float prev_energy;
} WaveletScale;

struct WaveletDetector {
  int sample_rate;
  int num_scales;
  WaveletScale *scales;

  void *(*alloc_fn)(size_t);
  void (*free_fn)(void *);

  // Global history (circular buffer) optimized for SIMD convolution
  float *global_history;
  int global_hist_idx;
  int max_kernel_size;
};

// Generate complex Morlet wavelet kernel
// psi(t) = pi^(-1/4) * exp(i*w0*t) * exp(-t^2/2)
static void generate_morlet_kernel(WaveletScale *ws, int sample_rate) {
  float w0 = 6.0f; // Standard frequency parameter for Morlet
  float dt = 1.0f / sample_rate;

  // Determine effective support of the wavelet (e.g., [-3sigma, 3sigma])
  // Scale 's' relates to frequency f via f = w0 / (2*pi*s) -> s = w0 / (2*pi*f)
  ws->scale = w0 / (2.0f * M_PI * ws->freq_hz);

  // Standard deviation in time domain is proportional to scale
  // We take a window of e.g. 6 * scale (buffer size)
  // Actually Morlet decay is exp(-t^2/2), so t=3 corresponds to significant
  // decay t is normalized by scale: exp(-(t/s)^2/2)
  float duration = 6.0f * ws->scale;
  ws->kernel_size = (int)(duration * sample_rate);

  // Ensure odd size for symmetry
  if (ws->kernel_size % 2 == 0)
    ws->kernel_size++;
  if (ws->kernel_size > MAX_KERNEL_SIZE)
    ws->kernel_size = MAX_KERNEL_SIZE;
  if (ws->kernel_size < 5)
    ws->kernel_size = 5;

  ws->kernel = (ComplexFloat *)malloc(sizeof(ComplexFloat) * ws->kernel_size);

  int center = ws->kernel_size / 2;
  float energy_norm = 0.0f;

  for (int i = 0; i < ws->kernel_size; i++) {
    float t = (i - center) * dt;
    float t_scaled = t / ws->scale;

    // Gaussian window
    float envelope = expf(-0.5f * t_scaled * t_scaled);

    // Complex sinusoid: exp(i * 2*pi * f * t) = cos(...) + i*sin(...)
    float phase = 2.0f * M_PI * ws->freq_hz * t;

    ws->kernel[i].r = envelope * cosf(phase);
    ws->kernel[i].i = envelope * sinf(phase);

    energy_norm +=
        ws->kernel[i].r * ws->kernel[i].r + ws->kernel[i].i * ws->kernel[i].i;
  }

  // Normalize kernel energy to 1
  energy_norm = sqrtf(energy_norm);
  for (int i = 0; i < ws->kernel_size; i++) {
    ws->kernel[i].r /= energy_norm;
    ws->kernel[i].i /= energy_norm;
  }

  // Allocate history
  ws->input_history = (float *)calloc(ws->kernel_size, sizeof(float));
  ws->history_idx = 0;
}

static void *default_malloc(size_t size) { return malloc(size); }
static void default_free(void *ptr) { free(ptr); }

WaveletDetector *wavelet_create(int sample_rate, float min_freq, float max_freq,
                                int num_scales, void *(*alloc_fn)(size_t)) {
  if (!alloc_fn)
    alloc_fn = default_malloc;

  WaveletDetector *wd = (WaveletDetector *)alloc_fn(sizeof(WaveletDetector));
  if (!wd)
    return NULL;

  wd->sample_rate = sample_rate;
  wd->num_scales = num_scales;
  wd->alloc_fn = alloc_fn;
  wd->free_fn =
      default_free; // Should use user free if provided, but strict API doesn't
                    // pass it. Assuming std free for internal allocs if user
                    // alloc is NULL is safer, but mixed is bad.
  // Improvement: Design assumes implementation handles sub-allocations.
  // We will use standard malloc/free for internal structures to avoid
  // complexity if user doesn't provide free_fn paired with alloc_fn. Actually,
  // let's just use standard malloc/free for internals for simplicity unless
  // critical.

  wd->scales = (WaveletScale *)calloc(num_scales, sizeof(WaveletScale));

  wd->max_kernel_size = 0;

  // Logarithmic frequency spacing
  float log_min = logf(min_freq);
  float log_max = logf(max_freq);
  float log_step = (log_max - log_min) / (num_scales > 1 ? num_scales - 1 : 1);

  for (int i = 0; i < num_scales; i++) {
    float freq = expf(log_min + i * log_step);
    wd->scales[i].freq_hz = freq;
    generate_morlet_kernel(&wd->scales[i], sample_rate);

    if (wd->scales[i].kernel_size > wd->max_kernel_size) {
      wd->max_kernel_size = wd->scales[i].kernel_size;
    }
  }

  // Prepare global history buffer
  wd->global_history = (float *)calloc(wd->max_kernel_size * 2, sizeof(float));
  wd->global_hist_idx = 0;

  return wd;
}

void wavelet_destroy(WaveletDetector *wd, void (*free_fn)(void *)) {
  if (!wd)
    return;

  for (int i = 0; i < wd->num_scales; i++) {
    free(wd->scales[i].kernel);
    free(wd->scales[i].input_history);
  }
  free(wd->scales);
  free(wd->global_history);

  if (free_fn)
    free_fn(wd);
  else
    free(wd);
}

void wavelet_reset(WaveletDetector *wd) {
  for (int i = 0; i < wd->num_scales; i++) {
    memset(wd->scales[i].input_history, 0,
           wd->scales[i].kernel_size * sizeof(float));
    wd->scales[i].history_idx = 0;
    wd->scales[i].current_magnitude = 0.0f;
    wd->scales[i].current_energy = 0.0f;
    wd->scales[i].prev_energy = 0.0f;
  }
  memset(wd->global_history, 0, wd->max_kernel_size * 2 * sizeof(float));
  wd->global_hist_idx = 0;
}

// Convolution logic
// Returns energy of the wavelet response
float wavelet_process_scale(WaveletScale *ws, float new_sample) {
  // Update history buffer (circular)
  ws->input_history[ws->history_idx] = new_sample;
  int current_idx = ws->history_idx;
  ws->history_idx = (ws->history_idx + 1) % ws->kernel_size;

  // Convolution: sum(x[n] * kernel[K-1-n])
  // x[0] is newest sample (at current_idx)

  float r_sum = 0.0f;
  float i_sum = 0.0f;

  int k_size = ws->kernel_size;

  // Basic scalar convolution (can be SIMD optimized)
  for (int k = 0; k < k_size; k++) {
    // Access history in reverse time (newest to oldest)
    int h_idx = current_idx - k;
    if (h_idx < 0)
      h_idx += k_size;

    float val = ws->input_history[h_idx];
    r_sum += val * ws->kernel[k].r;
    i_sum += val * ws->kernel[k].i;
  }

  float magnitude = sqrtf(r_sum * r_sum + i_sum * i_sum);

  ws->prev_energy = ws->current_energy;
  ws->current_magnitude = magnitude;
  ws->current_energy = magnitude * magnitude;

  return ws->current_energy;
}

float wavelet_process(WaveletDetector *wd, float sample) {
  float total_transient_score = 0.0f;
  int vote_count = 0;

  // Process each scale
  for (int i = 0; i < wd->num_scales; i++) {
    float energy = wavelet_process_scale(&wd->scales[i], sample);

    // Transient detection logic:
    // Look for rapid increase in energy (spectral flux in wavelet domain)
    // dE/dt > threshold

    float diff = energy - wd->scales[i].prev_energy;
    if (diff > 0) {
      // Normalize by previous energy (Weber's Law) - Robustness Feature
      // To avoid div by zero, add epsilon
      float relative_change = diff / (wd->scales[i].prev_energy + 1e-6f);

      // Weight high frequencies more for transients?
      // Or just sum relative changes
      total_transient_score += relative_change;
      vote_count++;
    }
  }

  // Average relative change score
  if (vote_count > 0) {
    return total_transient_score / wd->num_scales; // Normalize by total scales
  }
  return 0.0f;
}

float wavelet_get_energy(WaveletDetector *wd, int scale_idx) {
  if (scale_idx >= 0 && scale_idx < wd->num_scales) {
    return wd->scales[scale_idx].current_energy;
  }
  return 0.0f;
}
