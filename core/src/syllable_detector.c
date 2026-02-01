/*
 * syllable_detector.c - Multi-Feature Syllable/Accent Detection
 *
 * Enhanced version with:
 *   - Original PeakRate + ZFF pipeline
 *   - Spectral Flux for onset detection
 *   - High-frequency energy for consonant detection
 *   - MFCC Delta for phoneme boundaries
 *   - Feature Fusion with weighted combination
 *   - Unvoiced onset detection path
 */

#include "syllable_detector.h"
#include "dsp/agc.h"
#include "dsp/biquad.h"
#include "dsp/envelope.h"
#include "dsp/high_freq_energy.h"
#include "dsp/mfcc.h"
#include "dsp/spectral_flux.h"
#include "dsp/wavelet.h"
#include "dsp/zff.h"
#include <float.h>
#include <math.h>
#include <stdio.h> // For debug printf
#include <stdlib.h>
#include <string.h>

// --- Constants & Defaults ---
#define DEFAULT_SAMPLE_RATE 44100
#define PROMINENCE_BUFFER_SIZE 16 // Power of 2
#define SILENCE_THRESHOLD 0.001f
#define FEATURE_HISTORY_SIZE 32 // For feature normalization
#define FUSION_HISTORY_SIZE 64  // For online threshold (median+MAD)

// Real-Time Mode Constants
#define RT_NUM_FEATURES 6
#define RT_BUF_SIZE 100
#define RT_MIN_THRESH 1e-9f

// --- Internal Structs ---

typedef struct {
  SyllableEvent event;
  int is_ready;
} BufferedEvent;

// Feature statistics for normalization
typedef struct {
  float mean;
  float var;
  float max_val;
  float alpha;      // EMA coefficient
  int sample_count; // For confidence estimation
} FeatureStats;

// Real-Time Calibration State
typedef struct {
  int is_calibrating;
  int sample_count;
  int target_samples;
  float buf[RT_NUM_FEATURES][RT_BUF_SIZE];
  int buf_idx;
  float gamma;
  float thresh[RT_NUM_FEATURES];
} RealtimeCalibration;

struct SyllableDetector {
  SyllableConfig config;
  uint64_t total_samples;

  // DSP Modules (Legacy)
  Biquad bp_filter;
  EnvelopeFollower env_follower;
  ZFF zff;

  // DSP Modules (NEW - Multi-Feature)
  SpectralFlux *spectral_flux;
  HighFreqEnergy *high_freq_energy;
  MFCC *mfcc;
  WaveletDetector *wavelet;
  AgcState *agc;

  // PeakRate State
  float prev_env;
  float current_peak_rate;
  float peak_rate_accum;

  // ZFF State
  float last_zff_val;
  float last_zff_slope;
  int last_epoch_samples_ago;
  float current_f0;
  float smoothed_f0;       // EMA-smoothed F0
  float prev_smoothed_f0;  // Previous frame F0 (for derivative)
  float f0_derivative;     // Frame-to-frame F0 change
  float min_f0_since_peak; // Minimum F0 since last peak (for rise detection)
  int f0_has_risen;        // Flag: F0 has risen since last peak
  int f0_jump_counter;     // For outlier rejection

  // F0 baseline for absolute level comparison (secondary accent)
  float f0_baseline;       // Running median approximation of F0
  float f0_baseline_alpha; // EMA coefficient for baseline
  float f0_semitone_diff;  // Current F0 - baseline in semitones

  // Online threshold: fusion score history for median+MAD
  float fusion_history[FUSION_HISTORY_SIZE];
  int fusion_history_idx;
  int fusion_history_count;
  float fusion_median; // Running median
  float fusion_mad;    // Median Absolute Deviation

  int voicing_counter;
  int unvoiced_counter;
  int is_voiced;
  int voiced_hold_samples;

  // Energy tracking (for gating)
  float current_energy; // Current envelope energy
  float energy_floor;   // Adaptive noise floor

  // TEO (Teager Energy Operator) - nonlinear energy for "forcefulness"
  float prev_sample;      // x[n-1] for TEO
  float prev_prev_sample; // x[n-2] for TEO (actually x[n+1] relative to prev)
  float current_teo;      // Current TEO value
  float teo_mean;         // Running mean for normalization
  float teo_var;          // Running variance

  // LER (Local Energy Ratio) - local vs long-term energy
  float short_energy; // Short-term energy (EMA, ~20ms)
  float long_energy;  // Long-term energy (EMA, ~500ms)
  float current_ler;  // Short/Long ratio

  // Adaptive PeakRate Threshold
  int adaptive_enabled;
  float adaptive_mean;
  float adaptive_var;
  float adaptive_alpha;

  // Feature Statistics (for normalization)
  FeatureStats stats_peak_rate;
  FeatureStats stats_spectral_flux;
  FeatureStats stats_high_freq;
  FeatureStats stats_mfcc_delta;
  FeatureStats stats_wavelet;

  // Current feature values (updated each sample/hop)
  float current_spectral_flux;
  float current_high_freq_energy;
  float current_mfcc_delta;
  float current_wavelet_score;
  float current_fusion_score;

  // State Machine
  enum {
    STATE_IDLE,
    STATE_ONSET_RISING, // PeakRate/FusionScore rising
    STATE_NUCLEUS,      // Valid syllable nucleus
    STATE_COOLDOWN      // Enforcing min distance
  } state;

  int state_timer;              // Samples in current state
  int max_onset_rising_samples; // NEW: Time limit for ONSET_RISING state

  // WIP Event (while building a syllable)
  SyllableEvent wip_event;
  float max_peak_rate_in_syllable;
  float max_fusion_score_in_syllable;
  float energy_accum;
  uint64_t onset_timestamp;
  uint64_t last_event_samples; // Time of last emitted event (for F0 bypass)
  int peak_sample_offset;
  SyllableOnsetType current_onset_type;

  // Event Buffer (Ring Buffer) for Prominence Context
  BufferedEvent event_buffer[PROMINENCE_BUFFER_SIZE];
  int buf_write_idx;
  int buf_read_idx;
  int buf_count;

  // Real-Time Calibration State
  RealtimeCalibration rt_cal;

  // Memory
  void *(*alloc_fn)(size_t);
  void (*free_fn)(void *);
};

// --- Helpers ---

static void *default_malloc(size_t size) { return malloc(size); }
static void default_free(void *ptr) { free(ptr); }

static float bandpass_center_hz(const SyllableConfig *cfg) {
  return (cfg->peak_rate_band_min + cfg->peak_rate_band_max) * 0.5f;
}

static float bandpass_q_factor(const SyllableConfig *cfg) {
  float bandwidth = cfg->peak_rate_band_max - cfg->peak_rate_band_min;
  if (bandwidth < 1.0f)
    bandwidth = 1.0f;
  return bandpass_center_hz(cfg) / bandwidth;
}

static void configure_bandpass(Biquad *filter, const SyllableConfig *cfg) {
  float q = bandpass_q_factor(cfg);
  if (q < 0.1f)
    q = 0.1f;
  biquad_config_bandpass(filter, (float)cfg->sample_rate,
                         bandpass_center_hz(cfg), q);
}

// Initialize feature statistics
static void init_feature_stats(FeatureStats *stats, float tau_ms,
                               int sample_rate) {
  stats->mean = 0.0f;
  stats->var = 0.0f;
  stats->max_val = 0.0f;
  stats->sample_count = 0;
  float tau_s = tau_ms * 0.001f;
  stats->alpha = 1.0f / (tau_s * sample_rate);
  if (stats->alpha > 1.0f)
    stats->alpha = 1.0f;
}

// --- Real-Time Mode Functions ---

// Fast log2 approximation (error < 3%)
static inline float fast_log2(float x) {
  union {
    float f;
    uint32_t i;
  } u = {x};
  return (float)(u.i - 1064866805) * 8.262958e-8f;
}

// Fast exp2 approximation
static inline float fast_exp2(float x) {
  union {
    float f;
    uint32_t i;
  } u;
  u.i = (uint32_t)((x + 126.94269504f) * 8388608.0f);
  return u.f;
}

// Fast geometric mean
static inline float fast_geo_mean(float log_sum, int n) {
  return fast_exp2(log_sum / n * 1.442695f); // log2(e)
}

// 50th percentile (median) using insertion sort - robust to outliers
static float percentile_50(const float *values, int n) {
  if (n < 5)
    return values[n > 0 ? n - 1 : 0];
  float sorted[RT_BUF_SIZE];
  memcpy(sorted, values, n * sizeof(float));
  for (int i = 1; i < n; i++) {
    float key = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }
  return sorted[(int)(n * 0.50f)]; // 50th percentile (median)
}

// Finalize calibration
static void finalize_rt_calibration(struct SyllableDetector *d) {
  RealtimeCalibration *cal = &d->rt_cal;
  int n = (cal->buf_idx < RT_BUF_SIZE) ? cal->buf_idx : RT_BUF_SIZE;

  cal->gamma = powf(10.0f, d->config.snr_threshold_db / 10.0f);

  // Calculate thresholds from calibration data: mean + gamma * std
  for (int k = 0; k < RT_NUM_FEATURES; k++) {
    if (n < 10) {
      // Not enough data, use conservative default
      cal->thresh[k] = 0.001f;
    } else {
      // Calculate mean and std from buffer
      float sum = 0.0f, sum_sq = 0.0f;
      for (int i = 0; i < n; i++) {
        sum += cal->buf[k][i];
        sum_sq += cal->buf[k][i] * cal->buf[k][i];
      }
      float mean = sum / n;
      float var = (sum_sq / n) - (mean * mean);
      float std = (var > 0) ? sqrtf(var) : 0.0f;

      // Threshold = mean + gamma * std (SNR-based)
      cal->thresh[k] = mean + cal->gamma * std;

      // Ensure minimum threshold to avoid division by zero
      if (cal->thresh[k] < 1e-6f) {
        cal->thresh[k] = 1e-6f;
      }
    }
  }
  cal->is_calibrating = 0;
}

// Update calibration buffers
static void update_rt_calibration(struct SyllableDetector *d) {
  RealtimeCalibration *cal = &d->rt_cal;
  int idx = cal->buf_idx % RT_BUF_SIZE;

  cal->buf[0][idx] = d->current_energy;
  cal->buf[1][idx] = d->current_peak_rate;
  cal->buf[2][idx] = d->current_spectral_flux;
  cal->buf[3][idx] = d->current_high_freq_energy;
  cal->buf[4][idx] = d->current_mfcc_delta;
  cal->buf[5][idx] = d->current_wavelet_score;

  cal->buf_idx++;
  cal->sample_count++;

  if (cal->sample_count >= cal->target_samples) {
    finalize_rt_calibration(d);
  }
}

// Compute real-time fusion score using geometric mean
static float compute_fusion_realtime(struct SyllableDetector *d) {
  RealtimeCalibration *cal = &d->rt_cal;

  if (cal->is_calibrating)
    return 0.0f;

  float f[RT_NUM_FEATURES] = {
      d->current_energy,        d->current_peak_rate,
      d->current_spectral_flux, d->current_high_freq_energy,
      d->current_mfcc_delta,    d->current_wavelet_score};

  int active = 0;
  float log_sum = 0.0f;
  float max_r = 0.0f;

  for (int k = 0; k < RT_NUM_FEATURES; k++) {
    float r = f[k] / cal->thresh[k];
    if (r > 1.0f) {
      active++;
      log_sum += logf(r);
    }
    if (r > max_r)
      max_r = r;
  }

  // Voicing confidence boost
  float voiced_conf = fminf(1.0f, (float)d->voicing_counter / 5.0f);
  if (voiced_conf > 0.5f) {
    active++;
    log_sum += logf(1.0f + voiced_conf);
  }

  if (active == 0)
    return 0.0f;

  // Geometric mean of ratios above threshold
  float geo_mean = expf(log_sum / active);

  // Normalize to [0, 1] range using sigmoid-like saturation
  // geo_mean = 1 -> 0.5, geo_mean = 2 -> ~0.73, geo_mean = 4 -> ~0.88
  float score = 1.0f - 1.0f / (1.0f + geo_mean * 0.5f);

  return score;
}

// Update feature statistics (Welford's algorithm with sample counting)
static void update_feature_stats(FeatureStats *stats, float value) {
  float delta = value - stats->mean;
  stats->mean += stats->alpha * delta;
  stats->var =
      (1.0f - stats->alpha) * (stats->var + stats->alpha * delta * delta);
  if (value > stats->max_val)
    stats->max_val = value;
  if (stats->sample_count < 100000) // Prevent overflow
    stats->sample_count++;
}

// Fast sigmoid approximation (for normalization)
static inline float fast_sigmoid(float x) {
  // Approximation: x / (1 + |x|)
  return x / (1.0f + fabsf(x));
}

// Get confidence based on sample count (0.0 to 1.0)
static inline float get_stats_confidence(const FeatureStats *stats,
                                         int sample_rate) {
  // Reach 90% confidence after ~500ms of samples
  int target_samples = sample_rate / 2; // 500ms
  if (stats->sample_count >= target_samples)
    return 1.0f;
  return (float)stats->sample_count / (float)target_samples;
}

// Get normalized value using Sigmoid (improved version)
// Returns value in [0, 1] range with better handling of extremes
static float normalize_feature_sigmoid(const FeatureStats *stats, float value,
                                       float *confidence) {
  float std = sqrtf(stats->var);

  // Low confidence if statistics aren't stable yet
  if (std < 1e-6f || stats->sample_count < 100) {
    if (confidence)
      *confidence = 0.1f;
    return 0.5f; // Neutral value
  }

  if (confidence) {
    *confidence = fminf(1.0f, (float)stats->sample_count / 1000.0f);
  }

  // Z-score
  float z = (value - stats->mean) / std;

  // Sigmoid mapping: [-inf, +inf] -> [0, 1]
  // Used offset: -1.0 to push mean values (z=0) to score ~0.27 instead of 0.5
  // This reduces false positives from background noise
  return (fast_sigmoid(z - 1.0f) + 1.0f) * 0.5f;
}

// Legacy normalized value (z-score, clamped) - kept for compatibility
static float normalize_feature(const FeatureStats *stats, float value) {
  float std = sqrtf(stats->var);
  if (std < 1e-6f)
    std = 1e-6f;
  float z = (value - stats->mean) / std;
  // Clamp to reasonable range and map to 0-1
  if (z < 0)
    z = 0;
  if (z > 4.0f)
    z = 4.0f;
  return z / 4.0f;
}

SyllableConfig syllable_default_config(int sample_rate) {
  SyllableConfig cfg;
  memset(&cfg, 0, sizeof(cfg));

  cfg.sample_rate = sample_rate > 0 ? sample_rate : DEFAULT_SAMPLE_RATE;
  cfg.zff_trend_window_ms = 10.0f;
  cfg.peak_rate_band_min = 500.0f;
  cfg.peak_rate_band_max = 3200.0f;
  cfg.min_syllable_dist_ms = 150.0f;
  cfg.threshold_peak_rate = 0.0003f;
  cfg.adaptive_peak_rate_k = 4.0f;
  cfg.adaptive_peak_rate_tau_ms = 500.0f;
  cfg.voiced_hold_ms = 30.0f;
  cfg.hysteresis_on_factor = 1.2f;
  cfg.hysteresis_off_factor = 0.8f;
  cfg.context_size = 2;

  // Multi-Feature Detection defaults
  cfg.enable_spectral_flux = 1;
  cfg.enable_high_freq_energy = 1;
  cfg.enable_mfcc_delta = 1;
  cfg.enable_wavelet = 1;
  cfg.enable_agc = 1;

  cfg.fft_size_ms = 32.0f;
  cfg.hop_size_ms = 16.0f;
  cfg.high_freq_cutoff_hz = 2000.0f;

  // Feature weights (tuned for balanced detection)
  cfg.weight_peak_rate = 0.30f;
  cfg.weight_spectral_flux = 0.25f;
  cfg.weight_high_freq = 0.15f;
  cfg.weight_mfcc_delta = 0.10f;
  cfg.weight_wavelet = 0.20f;
  cfg.weight_voiced_bonus = 0.10f;

  // Fusion blend ratio (alpha * max + (1-alpha) * avg)
  cfg.fusion_blend_alpha = 0.6f;

  cfg.unvoiced_onset_threshold = 0.5f;
  cfg.allow_unvoiced_onsets = 1;

  // Real-Time Mode defaults
  cfg.realtime_mode = 0; // Default: offline mode
  cfg.calibration_duration_ms = 2000.0f;
  cfg.snr_threshold_db = 6.0f;

  // Hierarchical Prominence defaults (NEW)
  // Duration as continuous feature (Fry 1955)
  cfg.duration_weight = 0.1f;     // Weight from Python optimization
  cfg.duration_center_ms = 80.0f; // Optimal center from sweep
  cfg.duration_scale_ms = 30.0f;  // Sigmoid steepness

  // Pause-based phrase detection
  cfg.pause_threshold_ms = 200.0f;  // 200ms pause = phrase boundary
  cfg.phrase_initial_boost = 0.15f; // Boost for phrase-initial syllables

  cfg.user_malloc = NULL;
  cfg.user_free = NULL;

  return cfg;
}

// Calculate delta_f0: difference from context median
static void calculate_delta_f0(SyllableDetector *d, int target_idx) {
  BufferedEvent *target = &d->event_buffer[target_idx];
  if (!target->is_ready) {
    target->event.delta_f0 = 0.0f;
    return;
  }

  // Gather F0 values from context
  float f0_values[PROMINENCE_BUFFER_SIZE];
  int f0_count = 0;

  for (int i = 1; i <= d->config.context_size; i++) {
    int prev_idx =
        (target_idx - i + PROMINENCE_BUFFER_SIZE) % PROMINENCE_BUFFER_SIZE;
    if (d->event_buffer[prev_idx].is_ready &&
        d->event_buffer[prev_idx].event.f0 > 50.0f) {
      f0_values[f0_count++] = d->event_buffer[prev_idx].event.f0;
    }
    int next_idx = (target_idx + i) % PROMINENCE_BUFFER_SIZE;
    if (d->event_buffer[next_idx].is_ready &&
        d->event_buffer[next_idx].event.f0 > 50.0f) {
      f0_values[f0_count++] = d->event_buffer[next_idx].event.f0;
    }
  }

  if (f0_count == 0 || target->event.f0 < 50.0f) {
    target->event.delta_f0 = 0.0f;
    return;
  }

  // Simple median approximation: sort and take middle
  for (int i = 0; i < f0_count - 1; i++) {
    for (int j = 0; j < f0_count - i - 1; j++) {
      if (f0_values[j] > f0_values[j + 1]) {
        float tmp = f0_values[j];
        f0_values[j] = f0_values[j + 1];
        f0_values[j + 1] = tmp;
      }
    }
  }
  float median_f0 = f0_values[f0_count / 2];
  target->event.delta_f0 = target->event.f0 - median_f0;
}

static float calculate_prominence(SyllableDetector *d, int target_idx) {
  BufferedEvent *target = &d->event_buffer[target_idx];
  if (!target->is_ready)
    return 0.0f;

  float local_avg_energy = 0.0f;
  float local_avg_pr = 0.0f;
  float local_avg_dur = 0.0f;
  float local_avg_slope = 0.0f;
  float local_avg_fusion = 0.0f;
  int count = 0;

  // Gather context values
  for (int i = 1; i <= d->config.context_size; i++) {
    int prev_idx =
        (target_idx - i + PROMINENCE_BUFFER_SIZE) % PROMINENCE_BUFFER_SIZE;
    if (d->event_buffer[prev_idx].is_ready) {
      local_avg_energy += d->event_buffer[prev_idx].event.energy;
      local_avg_pr += d->event_buffer[prev_idx].event.peak_rate;
      local_avg_dur += d->event_buffer[prev_idx].event.duration_s;
      local_avg_slope += d->event_buffer[prev_idx].event.pr_slope;
      local_avg_fusion += d->event_buffer[prev_idx].event.fusion_score;
      count++;
    }

    int next_idx = (target_idx + i) % PROMINENCE_BUFFER_SIZE;
    if (d->event_buffer[next_idx].is_ready) {
      local_avg_energy += d->event_buffer[next_idx].event.energy;
      local_avg_pr += d->event_buffer[next_idx].event.peak_rate;
      local_avg_dur += d->event_buffer[next_idx].event.duration_s;
      local_avg_slope += d->event_buffer[next_idx].event.pr_slope;
      local_avg_fusion += d->event_buffer[next_idx].event.fusion_score;
      count++;
    }
  }

  if (count == 0)
    return 0.5f;

  local_avg_energy /= count;
  local_avg_pr /= count;
  local_avg_dur /= count;
  local_avg_slope /= count;
  local_avg_fusion /= count;

  // Calculate individual ratio scores
  float e_score = (target->event.energy > 0)
                      ? (target->event.energy / (local_avg_energy + 0.0001f))
                      : 0.0f;
  float pr_score = (target->event.peak_rate > 0)
                       ? (target->event.peak_rate / (local_avg_pr + 0.0001f))
                       : 0.0f;
  float d_score = (target->event.duration_s > 0)
                      ? (target->event.duration_s / (local_avg_dur + 0.0001f))
                      : 0.0f;
  float slope_score =
      (target->event.pr_slope > 0)
          ? (target->event.pr_slope / (local_avg_slope + 0.0001f))
          : 0.0f;
  float fusion_score_ratio =
      (target->event.fusion_score > 0)
          ? (target->event.fusion_score / (local_avg_fusion + 0.0001f))
          : 0.0f;

  // F0 change bonus
  float f0_bonus =
      (target->event.delta_f0 > 0) ? (target->event.delta_f0 / 50.0f) : 0.0f;
  if (f0_bonus > 1.0f)
    f0_bonus = 1.0f;

  // Stress integral: FusionScore × Duration (key Weber-Fechner metric)
  // This captures "how strong and how long" the syllable is
  float stress_integral = target->event.fusion_score * target->event.duration_s;
  float avg_stress = local_avg_fusion * local_avg_dur;
  float stress_ratio =
      (avg_stress > 0.001f) ? (stress_integral / avg_stress) : 1.0f;
  if (stress_ratio > 3.0f)
    stress_ratio = 3.0f; // Clamp for phrase-final

  // F0 absolute level bonus (for secondary accents with high pitch but low
  // energy) Positive semitone difference = pitch higher than baseline =
  // potential accent Bonus kicks in at 2+ semitones above baseline
  float f0_level_bonus = 0.0f;
  if (target->event.f0 > 60.0f) { // Valid F0
    // Approximate semitone diff from stored F0 and typical baseline
    // This is a simplified version; ideally we'd store f0_semitone_diff per
    // event
    float f0_norm = target->event.f0 / 150.0f;  // Normalize around typical F0
    if (f0_norm > 1.1f) {                       // Higher than average
      f0_level_bonus = (f0_norm - 1.0f) * 0.5f; // Scale to 0-0.15 range
      if (f0_level_bonus > 0.15f)
        f0_level_bonus = 0.15f;
    }
  }

  // --- Hierarchical Prominence (NEW) ---
  // 1. Duration as continuous feature (Fry 1955)
  //    Formula: weight = sigmoid((duration_ms - center) / scale)
  //    This smoothly modulates score based on syllable length
  float duration_ms = target->event.duration_s * 1000.0f;
  float dur_z = (duration_ms - d->config.duration_center_ms) /
                d->config.duration_scale_ms;
  float duration_sigmoid = 1.0f / (1.0f + expf(-dur_z));

  // 2. Phrase-initial detection (based on pause before syllable)
  //    If previous syllable is far away, this might be phrase-initial
  //    NOTE: First syllable is NOT boosted (no context to compare)
  float phrase_boost = 0.0f;
  int prev_idx =
      (target_idx - 1 + PROMINENCE_BUFFER_SIZE) % PROMINENCE_BUFFER_SIZE;
  if (d->event_buffer[prev_idx].is_ready) {
    float gap_ms = (target->event.time_seconds -
                    d->event_buffer[prev_idx].event.time_seconds) *
                       1000.0f -
                   d->event_buffer[prev_idx].event.duration_s * 1000.0f;
    if (gap_ms > d->config.pause_threshold_ms) {
      phrase_boost = d->config.phrase_initial_boost;
    }
  }
  // First syllable: no boost (removed to avoid false positives)

  // Combined score with enhanced duration weight and F0 level
  // Duration is the most dominant parameter for stress perception
  float base_score = 0.10f * e_score + 0.10f * pr_score +
                     0.18f * d_score + // Duration weight
                     0.08f * slope_score + 0.18f * fusion_score_ratio +
                     0.13f * stress_ratio + 0.10f * (1.0f + f0_bonus) +
                     0.13f * (1.0f + f0_level_bonus); // F0 absolute level

  // Apply duration modulation: score * (1 - w + w * sigmoid)
  // This preserves score but modulates based on duration
  float w = d->config.duration_weight;
  float score = base_score * (1.0f - w + w * duration_sigmoid) + phrase_boost;

  return score;
}

// --- API Implementation ---

SyllableDetector *syllable_create(const SyllableConfig *config) {
  SyllableConfig cfg =
      config ? *config : syllable_default_config(DEFAULT_SAMPLE_RATE);

  void *(*alloc)(size_t) = cfg.user_malloc ? cfg.user_malloc : default_malloc;

  SyllableDetector *d = (SyllableDetector *)alloc(sizeof(SyllableDetector));
  if (!d)
    return NULL;

  memset(d, 0, sizeof(SyllableDetector));
  d->config = cfg;
  d->alloc_fn = alloc;
  d->free_fn = cfg.user_free ? cfg.user_free : default_free;

  // Init Legacy DSP
  biquad_reset(&d->bp_filter);
  configure_bandpass(&d->bp_filter, &cfg);

  envelope_init(&d->env_follower, (float)cfg.sample_rate, 5.0f, 20.0f);

  zff_init(&d->zff, cfg.sample_rate, cfg.zff_trend_window_ms, d->alloc_fn);

  d->voiced_hold_samples = (int)(cfg.voiced_hold_ms * 0.001f * cfg.sample_rate);
  if (d->voiced_hold_samples < 1)
    d->voiced_hold_samples = 1;

  // IMPROVED: Set max time for ONSET_RISING state (50ms)
  d->max_onset_rising_samples = (int)(0.050f * cfg.sample_rate);

  // Initialize F0 smoothing and energy tracking
  d->smoothed_f0 = 0.0f;
  d->prev_smoothed_f0 = 0.0f;
  d->f0_derivative = 0.0f;
  d->min_f0_since_peak = 0.0f;
  d->f0_has_risen = 1; // Start as true to allow first detection
  d->f0_jump_counter = 0;
  d->current_energy = 0.0f;
  d->energy_floor = 0.0f;

  // Initialize TEO
  d->prev_sample = 0.0f;
  d->prev_prev_sample = 0.0f;
  d->current_teo = 0.0f;
  d->teo_mean = 0.0f;
  d->teo_var = 0.0f;

  // Initialize LER with EMA coefficients
  // Short: ~20ms, Long: ~500ms
  d->short_energy = 0.0f;
  d->long_energy = 0.0001f; // Small value to avoid division by zero
  d->current_ler = 1.0f;

  // Initialize F0 baseline (for secondary accent detection)
  d->f0_baseline = 0.0f;
  d->f0_baseline_alpha = 1.0f - expf(-1.0f / (1.0f * cfg.sample_rate)); // ~1s
  d->f0_semitone_diff = 0.0f;

  // Initialize online threshold (fusion score history)
  memset(d->fusion_history, 0, sizeof(d->fusion_history));
  d->fusion_history_idx = 0;
  d->fusion_history_count = 0;
  d->fusion_median = 0.5f;
  d->fusion_mad = 0.2f;

  // Init Multi-Feature DSP
  int fft_size = (int)(cfg.fft_size_ms * 0.001f * cfg.sample_rate);
  // Round to power of 2
  int fft_power = 1;
  while (fft_power < fft_size)
    fft_power <<= 1;
  fft_size = fft_power;

  int hop_size = (int)(cfg.hop_size_ms * 0.001f * cfg.sample_rate);

  if (cfg.enable_spectral_flux) {
    d->spectral_flux =
        spectral_flux_create(cfg.sample_rate, fft_size, hop_size, alloc);
  }

  if (cfg.enable_high_freq_energy) {
    d->high_freq_energy =
        hfe_create(cfg.sample_rate, cfg.high_freq_cutoff_hz, 10.0f, alloc);
  }

  if (cfg.enable_mfcc_delta) {
    d->mfcc = mfcc_create(cfg.sample_rate, fft_size, hop_size, alloc);
  }

  if (cfg.enable_wavelet) {
    // 3 scales from 2000Hz to 6000Hz for high-frequency transients
    d->wavelet = wavelet_create(cfg.sample_rate, 2000.0f, 6000.0f, 3, alloc);
  }

  if (cfg.enable_agc) {
    // Target -23dB (broadcast standard), max gain 30dB
    d->agc = agc_create(cfg.sample_rate, -23.0f, 30.0f, alloc);
  }

  // Adaptive threshold
  d->adaptive_enabled =
      (cfg.adaptive_peak_rate_k > 0.0f && cfg.adaptive_peak_rate_tau_ms > 0.0f);
  if (d->adaptive_enabled) {
    float tau_s = cfg.adaptive_peak_rate_tau_ms * 0.001f;
    d->adaptive_alpha = 1.0f / (tau_s * cfg.sample_rate);
    if (d->adaptive_alpha > 1.0f)
      d->adaptive_alpha = 1.0f;
  } else {
    d->adaptive_alpha = 0.0f;
  }

  // Init feature statistics
  init_feature_stats(&d->stats_peak_rate, cfg.adaptive_peak_rate_tau_ms,
                     cfg.sample_rate);
  init_feature_stats(&d->stats_spectral_flux, cfg.adaptive_peak_rate_tau_ms,
                     cfg.sample_rate);
  init_feature_stats(&d->stats_high_freq, cfg.adaptive_peak_rate_tau_ms,
                     cfg.sample_rate);
  init_feature_stats(&d->stats_mfcc_delta, cfg.adaptive_peak_rate_tau_ms,
                     cfg.sample_rate);

  syllable_reset(d);

  return d;
}

void syllable_reset(SyllableDetector *d) {
  d->total_samples = 0;
  d->prev_env = 0.0f;
  d->state = STATE_IDLE;
  d->is_voiced = 0;
  d->voicing_counter = 0;
  d->unvoiced_counter = 0;
  d->buf_read_idx = 0;
  d->buf_write_idx = 0;
  d->buf_count = 0;

  // Clear buffer flags
  for (int i = 0; i < PROMINENCE_BUFFER_SIZE; i++)
    d->event_buffer[i].is_ready = 0;

  // Reset Legacy DSP
  biquad_reset(&d->bp_filter);
  configure_bandpass(&d->bp_filter, &d->config);
  d->env_follower.output = 0.0f;
  if (d->zff.trend_buffer && d->zff.trend_buf_size > 0) {
    memset(d->zff.trend_buffer, 0,
           (size_t)d->zff.trend_buf_size * sizeof(float));
  }
  d->zff.int1 = 0.0;
  d->zff.int2 = 0.0;
  d->zff.trend_write_pos = 0;
  d->zff.trend_accum = 0.0f;

  d->adaptive_mean = 0.0f;
  d->adaptive_var = 0.0f;

  // Reset Multi-Feature DSP
  if (d->spectral_flux)
    spectral_flux_reset(d->spectral_flux);
  if (d->high_freq_energy)
    hfe_reset(d->high_freq_energy);
  if (d->mfcc)
    mfcc_reset(d->mfcc);

  // Reset feature values
  d->current_spectral_flux = 0.0f;
  d->current_high_freq_energy = 0.0f;
  d->current_mfcc_delta = 0.0f;
  d->current_fusion_score = 0.0f;

  // Reset feature stats
  init_feature_stats(&d->stats_peak_rate, d->config.adaptive_peak_rate_tau_ms,
                     d->config.sample_rate);
  init_feature_stats(&d->stats_spectral_flux,
                     d->config.adaptive_peak_rate_tau_ms,
                     d->config.sample_rate);
  init_feature_stats(&d->stats_high_freq, d->config.adaptive_peak_rate_tau_ms,
                     d->config.sample_rate);
  init_feature_stats(&d->stats_mfcc_delta, d->config.adaptive_peak_rate_tau_ms,
                     d->config.sample_rate);
  init_feature_stats(&d->stats_wavelet, d->config.adaptive_peak_rate_tau_ms,
                     d->config.sample_rate);
}

void syllable_destroy(SyllableDetector *d) {
  if (!d)
    return;

  // Destroy Multi-Feature DSP
  if (d->spectral_flux)
    spectral_flux_destroy(d->spectral_flux, d->free_fn);
  if (d->high_freq_energy)
    hfe_destroy(d->high_freq_energy, d->free_fn);
  if (d->mfcc)
    mfcc_destroy(d->mfcc, d->free_fn);
  if (d->wavelet)
    wavelet_destroy(d->wavelet, d->free_fn);
  if (d->agc)
    agc_destroy(d->agc, d->free_fn);

  zff_destroy(&d->zff, d->free_fn);
  d->free_fn(d);
}

// Compute fusion score from all features (IMPROVED: Energy-Gated + Max/Avg
// blend)
static float compute_fusion_score(SyllableDetector *d) {
  // Real-time mode: use the new geometric mean fusion
  if (d->config.realtime_mode) {
    return compute_fusion_realtime(d);
  }

  // Energy gating: if too close to noise floor, return reduced score
  // Use relative threshold: must be at least 3x the noise floor
  float energy_ratio = 1.0f;
  if (d->energy_floor > 1e-8f) {
    energy_ratio = d->current_energy / d->energy_floor;
  }

  // Very weak signal gate
  if (d->current_energy < 1e-6f || energy_ratio < 1.5f) {
    return 0.0f;
  }

  // Get normalized features using improved sigmoid normalization
  float conf_pr, conf_sf, conf_hf, conf_mfcc;
  float norm_pr = normalize_feature_sigmoid(&d->stats_peak_rate,
                                            d->current_peak_rate, &conf_pr);
  float norm_sf =
      d->spectral_flux
          ? normalize_feature_sigmoid(&d->stats_spectral_flux,
                                      d->current_spectral_flux, &conf_sf)
          : 0.0f;
  float norm_hf =
      d->high_freq_energy
          ? normalize_feature_sigmoid(&d->stats_high_freq,
                                      d->current_high_freq_energy, &conf_hf)
          : 0.0f;
  float norm_mfcc =
      d->mfcc ? normalize_feature_sigmoid(&d->stats_mfcc_delta,
                                          d->current_mfcc_delta, &conf_mfcc)
              : 0.0f;
  float norm_wavelet =
      d->wavelet ? normalize_feature_sigmoid(&d->stats_wavelet,
                                             d->current_wavelet_score, NULL)
                 : 0.0f;
  float voiced_bonus = d->is_voiced ? 1.0f : 0.0f;

  // Calculate weighted average (traditional)
  float w_total = d->config.weight_peak_rate;
  float weighted_avg = d->config.weight_peak_rate * norm_pr;

  if (d->spectral_flux) {
    weighted_avg += d->config.weight_spectral_flux * norm_sf;
    w_total += d->config.weight_spectral_flux;
  }
  if (d->high_freq_energy) {
    weighted_avg += d->config.weight_high_freq * norm_hf;
    w_total += d->config.weight_high_freq;
  }
  if (d->mfcc) {
    weighted_avg += d->config.weight_mfcc_delta * norm_mfcc;
    w_total += d->config.weight_mfcc_delta;
  }
  if (d->wavelet) {
    weighted_avg += d->config.weight_wavelet * norm_wavelet;
    w_total += d->config.weight_wavelet;
  }
  weighted_avg += d->config.weight_voiced_bonus * voiced_bonus;
  w_total += d->config.weight_voiced_bonus;

  if (w_total > 0)
    weighted_avg /= w_total;

  // Calculate max feature (for strong single-feature onsets)
  float max_feature = norm_pr;
  if (d->spectral_flux && norm_sf > max_feature)
    max_feature = norm_sf;
  if (d->high_freq_energy && norm_hf > max_feature)
    max_feature = norm_hf;
  if (d->mfcc && norm_mfcc > max_feature)
    max_feature = norm_mfcc;
  if (d->wavelet && norm_wavelet > max_feature)
    max_feature = norm_wavelet;

  // Blend: alpha * max + (1-alpha) * average (configurable for optimization)
  float alpha = d->config.fusion_blend_alpha;
  float fusion = alpha * max_feature + (1.0f - alpha) * weighted_avg;

  // Apply confidence weighting (reduce score if stats are unstable)
  float avg_confidence = conf_pr;
  int conf_count = 1;
  if (d->spectral_flux) {
    avg_confidence += conf_sf;
    conf_count++;
  }
  if (d->high_freq_energy) {
    avg_confidence += conf_hf;
    conf_count++;
  }
  if (d->mfcc) {
    avg_confidence += conf_mfcc;
    conf_count++;
  }
  // Wavelet adds to score but we don't track confidence for it yet
  avg_confidence /= conf_count;

  // Only apply confidence penalty if very low
  if (avg_confidence < 0.3f) {
    fusion *= (0.5f + avg_confidence); // Scale down uncertain detections
  }

  return fusion;
}

// Determine onset type based on features
static SyllableOnsetType determine_onset_type(SyllableDetector *d) {
  if (d->is_voiced) {
    // Check if high-freq energy is significant (possible fricative)
    float hf_norm = d->high_freq_energy
                        ? normalize_feature(&d->stats_high_freq,
                                            d->current_high_freq_energy)
                        : 0.0f;
    if (hf_norm > 0.5f) {
      return ONSET_TYPE_MIXED; // Voiced fricative
    }
    return ONSET_TYPE_VOICED;
  } else {
    return ONSET_TYPE_UNVOICED;
  }
}

int syllable_process(SyllableDetector *d, const float *input, int num_samples,
                     SyllableEvent *events_out, int max_events) {
  int events_written = 0;

  // Temporary buffers for frame-based features
  float flux_buf[8];
  float mfcc_delta_buf[8]; // Process sample by sample
  for (int i = 0; i < num_samples; i++) {
    float in_sample = input[i];

    // 0. AGC (Robustness)
    if (d->agc) {
      in_sample = agc_process(d->agc, in_sample);
    }

    d->total_samples++;

    // 1. ZFF Detection (Voicing / F0) with IMPROVED F0 smoothing
    float zff_out, zff_slope;
    zff_process(&d->zff, in_sample, &zff_out, &zff_slope);

    int is_epoch = 0;
    if (d->last_zff_val < 0.0f && zff_out >= 0.0f) {
      is_epoch = 1;

      if (d->last_epoch_samples_ago > 0) {
        float period_s =
            (float)d->last_epoch_samples_ago / d->config.sample_rate;
        float raw_f0 = 1.0f / period_s;

        // Validate F0 range (50-600 Hz for human voice)
        if (raw_f0 > 50 && raw_f0 < 600) {
          // IMPROVED: F0 smoothing with outlier rejection
          if (d->smoothed_f0 < 50.0f) {
            // First valid F0, initialize directly
            d->smoothed_f0 = raw_f0;
            d->current_f0 = raw_f0;
            d->f0_jump_counter = 0;
          } else {
            // Check if this is within acceptable range (20% deviation)
            float deviation = fabsf(raw_f0 - d->smoothed_f0) / d->smoothed_f0;
            if (deviation < 0.2f) {
              // Normal tracking: apply EMA
              d->smoothed_f0 = 0.7f * d->smoothed_f0 + 0.3f * raw_f0;
              d->current_f0 = d->smoothed_f0;
              d->f0_jump_counter = 0;
            } else {
              // Possible octave jump or noise - require confirmation
              d->f0_jump_counter++;
              if (d->f0_jump_counter > 3) {
                // Confirmed: accept the new F0 range
                d->smoothed_f0 = raw_f0;
                d->current_f0 = raw_f0;
                d->f0_jump_counter = 0;
              }
              // Otherwise keep using smoothed_f0
            }
          }
          d->voicing_counter = 5;
        }
      }
      d->last_epoch_samples_ago = 0;
    } else {
      d->last_epoch_samples_ago++;
    }

    d->last_zff_val = zff_out;

    // Voicing logic
    if (d->voicing_counter > 0)
      d->is_voiced = 1;
    else
      d->is_voiced = 0;

    if (!is_epoch) {
      if (d->last_epoch_samples_ago > d->voiced_hold_samples) {
        d->is_voiced = 0;
      } else {
        d->is_voiced = 1;
      }
    }

    // F0 derivative calculation (for detecting F0 rise = new syllable)
    d->f0_derivative = d->smoothed_f0 - d->prev_smoothed_f0;
    d->prev_smoothed_f0 = d->smoothed_f0;

    // Track minimum F0 since last peak (for rise detection)
    // Initialize min_f0 if it's 0 (first valid F0) or if current is lower
    if (d->smoothed_f0 > 50.0f) {
      if (d->min_f0_since_peak < 50.0f) {
        // First valid F0 since reset - initialize
        d->min_f0_since_peak = d->smoothed_f0;
      } else if (d->smoothed_f0 < d->min_f0_since_peak) {
        d->min_f0_since_peak = d->smoothed_f0;
      }

      // Check if F0 has risen significantly since last peak
      // Rise threshold: 5% above minimum
      if (d->smoothed_f0 > d->min_f0_since_peak * 1.05f) {
        d->f0_has_risen = 1;
      }

      // Update F0 baseline (slow EMA for local reference)
      if (d->f0_baseline < 50.0f) {
        d->f0_baseline = d->smoothed_f0; // Initialize
      } else {
        d->f0_baseline = d->f0_baseline_alpha * d->smoothed_f0 +
                         (1.0f - d->f0_baseline_alpha) * d->f0_baseline;
      }

      // Compute semitone difference from baseline
      // Positive = current F0 higher than baseline = potential accent
      d->f0_semitone_diff =
          12.0f * log2f(d->smoothed_f0 / (d->f0_baseline + 0.1f));

    } else {
      // No valid F0 (unvoiced segment) - allow detection based on other
      // features This ensures unvoiced consonants can still be detected
      if (!d->is_voiced) {
        d->f0_has_risen = 1; // Allow unvoiced onsets
      }
      d->f0_semitone_diff = 0.0f; // Reset for unvoiced
    }

    // 2. PeakRate Pipeline (Legacy) + Energy Tracking
    float bp_out = biquad_process(&d->bp_filter, in_sample);
    float env_out = envelope_process(&d->env_follower, bp_out);

    float diff = env_out - d->prev_env;
    float peak_rate = (diff > 0.0f) ? diff : 0.0f;
    d->prev_env = env_out;
    d->current_peak_rate = peak_rate;

    // Track current energy for gating
    d->current_energy = env_out;

    // Update adaptive energy floor (noise floor estimation)
    // Slow attack, faster release
    if (env_out < d->energy_floor || d->energy_floor < 1e-8f) {
      d->energy_floor = env_out; // Fast adaptation to lower values
    } else {
      d->energy_floor =
          0.9999f * d->energy_floor + 0.0001f * env_out; // Slow rise
    }

    // TEO (Teager Energy Operator): Ψ[x(n)] = x(n)² - x(n-1) * x(n+1)
    // We compute with delay: x[n-1]² - x[n-2] * x[n]
    float teo_raw =
        d->prev_sample * d->prev_sample - d->prev_prev_sample * in_sample;
    if (teo_raw < 0.0f)
      teo_raw = 0.0f; // Half-wave rectify
    d->current_teo = teo_raw;

    // Update TEO stats for normalization (EMA)
    float teo_alpha = 0.001f; // ~1000 samples
    float teo_delta = teo_raw - d->teo_mean;
    d->teo_mean += teo_alpha * teo_delta;
    d->teo_var =
        (1.0f - teo_alpha) * (d->teo_var + teo_alpha * teo_delta * teo_delta);

    // Shift samples for next TEO
    d->prev_prev_sample = d->prev_sample;
    d->prev_sample = in_sample;

    // LER (Local Energy Ratio): short-term / long-term energy
    // EMA coefficients: short ~20ms, long ~500ms
    float sample_energy = in_sample * in_sample;
    float alpha_short = 1.0f - expf(-1.0f / (0.020f * d->config.sample_rate));
    float alpha_long = 1.0f - expf(-1.0f / (0.500f * d->config.sample_rate));

    d->short_energy =
        alpha_short * sample_energy + (1.0f - alpha_short) * d->short_energy;
    d->long_energy =
        alpha_long * sample_energy + (1.0f - alpha_long) * d->long_energy;

    // Compute LER (clamped to reasonable range)
    if (d->long_energy > 1e-10f) {
      d->current_ler = d->short_energy / d->long_energy;
      if (d->current_ler > 10.0f)
        d->current_ler = 10.0f; // Clamp
    } else {
      d->current_ler = 1.0f;
    }

    // Update PeakRate stats
    if (d->is_voiced || d->config.allow_unvoiced_onsets) {
      update_feature_stats(&d->stats_peak_rate, peak_rate);
    }

    // 3. Multi-Feature Processing

    // Spectral Flux (frame-based, updates less frequently)
    if (d->spectral_flux) {
      int n_flux =
          spectral_flux_process(d->spectral_flux, &in_sample, 1, flux_buf, 8);
      if (n_flux > 0) {
        d->current_spectral_flux = flux_buf[n_flux - 1];
        update_feature_stats(&d->stats_spectral_flux, d->current_spectral_flux);
      }
    }

    // High-Frequency Energy (sample-based)
    if (d->high_freq_energy) {
      d->current_high_freq_energy = hfe_process(d->high_freq_energy, in_sample);
      update_feature_stats(&d->stats_high_freq, d->current_high_freq_energy);
    }

    // MFCC Delta (frame-based)
    if (d->mfcc) {
      int n_mfcc = mfcc_process(d->mfcc, &in_sample, 1, mfcc_delta_buf, 8);
      if (n_mfcc > 0) {
        d->current_mfcc_delta = mfcc_delta_buf[n_mfcc - 1];
        update_feature_stats(&d->stats_mfcc_delta, d->current_mfcc_delta);
      }
    }

    // Wavelet Transform (sample-based)
    if (d->wavelet) {
      d->current_wavelet_score = wavelet_process(d->wavelet, in_sample);
      update_feature_stats(&d->stats_wavelet, d->current_wavelet_score);
    }

    // Real-time mode calibration update
    if (d->config.realtime_mode && d->rt_cal.is_calibrating) {
      update_rt_calibration(d);
    }

    // 4. Compute Fusion Score
    d->current_fusion_score = compute_fusion_score(d);

    // Update fusion score history for online threshold (median + MAD)
    d->fusion_history[d->fusion_history_idx] = d->current_fusion_score;
    d->fusion_history_idx = (d->fusion_history_idx + 1) % FUSION_HISTORY_SIZE;
    if (d->fusion_history_count < FUSION_HISTORY_SIZE)
      d->fusion_history_count++;

    // Recompute median and MAD periodically (every 16 samples for efficiency)
    if ((d->total_samples % 16) == 0 && d->fusion_history_count >= 8) {
      // Simple approximation: use running statistics instead of exact median
      // For online processing, EMA-based approximation is sufficient
      float sum = 0.0f, sum_abs_dev = 0.0f;
      for (int j = 0; j < d->fusion_history_count; j++) {
        sum += d->fusion_history[j];
      }
      float mean = sum / d->fusion_history_count;
      for (int j = 0; j < d->fusion_history_count; j++) {
        float dev = d->fusion_history[j] - mean;
        sum_abs_dev += (dev > 0) ? dev : -dev;
      }
      d->fusion_median = mean; // Approximate median with mean
      d->fusion_mad = sum_abs_dev / d->fusion_history_count;
    }

    // Adaptive threshold for legacy path
    if (d->adaptive_enabled && d->is_voiced) {
      float delta = peak_rate - d->adaptive_mean;
      d->adaptive_mean += d->adaptive_alpha * delta;
      d->adaptive_var = (1.0f - d->adaptive_alpha) *
                        (d->adaptive_var + d->adaptive_alpha * delta * delta);
    }

    float threshold = d->config.threshold_peak_rate;
    if (d->adaptive_enabled) {
      float std = (d->adaptive_var > 0.0f) ? sqrtf(d->adaptive_var) : 0.0f;
      float adaptive = d->adaptive_mean + d->config.adaptive_peak_rate_k * std;
      if (adaptive > threshold)
        threshold = adaptive;
    }

    // Hysteresis thresholds
    float threshold_on = threshold * d->config.hysteresis_on_factor;
    float threshold_off = threshold * d->config.hysteresis_off_factor;

    // Fusion-based threshold (normalized, tuned for sigmoid output)
    // 0.6 corresponds to roughly mean + 1.4 sigma
    float fusion_threshold_on = 0.6f * d->config.hysteresis_on_factor;
    float fusion_threshold_off = 0.4f * d->config.hysteresis_off_factor;

    // 5. State Machine
    // SKIP state machine during realtime calibration to collect only noise
    // floor
    if (d->config.realtime_mode && d->rt_cal.is_calibrating) {
      // Only collect calibration data, no onset detection
      continue;
    }

    if (d->state == STATE_IDLE) {
      // Traditional voiced onset condition
      int voiced_trigger = (peak_rate > threshold_on && d->is_voiced);

      // Fusion-based trigger (can detect unvoiced onsets)
      int fusion_trigger = (d->current_fusion_score > fusion_threshold_on);

      // Unvoiced onset condition (high SF + HFE without voicing)
      int unvoiced_trigger = 0;
      if (d->config.allow_unvoiced_onsets && !d->is_voiced) {
        float sf_norm = d->spectral_flux
                            ? normalize_feature(&d->stats_spectral_flux,
                                                d->current_spectral_flux)
                            : 0.0f;
        float hf_norm = d->high_freq_energy
                            ? normalize_feature(&d->stats_high_freq,
                                                d->current_high_freq_energy)
                            : 0.0f;
        unvoiced_trigger = (sf_norm > d->config.unvoiced_onset_threshold ||
                            hf_norm > d->config.unvoiced_onset_threshold);
      }

      // NEW: F0 rise check - suppress detection if F0 hasn't risen since last
      // peak This prevents detecting echoes/continuations as new syllables
      //
      // HIERARCHICAL EVIDENCE: F0 condition can be bypassed by:
      // 1. F0 has actually risen (primary condition)
      // 2. Strong single-feature evidence (fusion score very high)
      // 3. Enough time has passed since last event (2x min_syllable_dist)

      int f0_condition = d->f0_has_risen;

      // Strong evidence bypass: if any Weber-Fechner saliency is high, trust it
      // TEO: normalized value > 3σ (nonlinear energy burst)
      float teo_std = (d->teo_var > 0) ? sqrtf(d->teo_var) : 1e-6f;
      float teo_normalized = (d->current_teo - d->teo_mean) / (teo_std + 1e-6f);
      int teo_strong = (teo_normalized > 3.0f); // 3σ above mean

      // LER: local energy is 2x higher than long-term (Weber ratio > 1)
      int ler_strong = (d->current_ler > 2.0f);

      // Spectral Flatness Weber: rapid harmonicity increase (vowel onset)
      // Negative Weber ratio means flatness decreased = becoming more harmonic
      float flatness_weber =
          d->spectral_flux ? spectral_flux_get_flatness_weber(d->spectral_flux)
                           : 0.0f;
      int harmonicity_strong =
          (flatness_weber < -0.3f); // 30% decrease in flatness

      int strong_evidence = (d->current_fusion_score > 0.85f) || teo_strong ||
                            ler_strong || harmonicity_strong;

      // Time-based bypass: if enough time passed, allow new detection
      int min_dist_samples = (int)(d->config.min_syllable_dist_ms * 0.001f *
                                   d->config.sample_rate);
      int time_since_last = (int)(d->total_samples - d->last_event_samples);
      int enough_time_passed = (time_since_last > min_dist_samples * 2);

      // Combined: any of these allows new onset
      // REALTIME FIX: In realtime mode, bypass F0 gate for immediate detection
      int f0_allows_new_onset =
          d->config.realtime_mode
              ? 1
              : (f0_condition || strong_evidence || enough_time_passed);

      // REALTIME FIX: Energy gate to prevent false positives from noise
      // Require current energy to significantly exceed calibrated threshold
      int energy_gate_passed = 1;
      if (d->config.realtime_mode && !d->rt_cal.is_calibrating) {
        // Energy must exceed calibrated noise floor by at least 3x
        // This provides ~10dB SNR margin above noise
        float energy_threshold = d->rt_cal.thresh[0] * 3.0f;

        // Also require minimum absolute energy to avoid triggering on quiet
        // rooms
        float min_absolute_energy = 0.001f; // Roughly -60dB

        energy_gate_passed = (d->current_energy > energy_threshold) &&
                             (d->current_energy > min_absolute_energy);
      }

      if ((voiced_trigger ||
           (fusion_trigger &&
            (d->config.allow_unvoiced_onsets || d->is_voiced)) ||
           unvoiced_trigger) &&
          f0_allows_new_onset && energy_gate_passed) {
        d->state = STATE_ONSET_RISING;
        d->state_timer = 0;

        // Determine onset type
        d->current_onset_type = determine_onset_type(d);

        // Init Event
        memset(&d->wip_event, 0, sizeof(d->wip_event));
        d->wip_event.timestamp_samples = d->total_samples;
        d->wip_event.time_seconds =
            (double)d->total_samples / d->config.sample_rate;
        d->wip_event.peak_rate = peak_rate;
        d->wip_event.pr_slope = 0.0f;
        d->wip_event.energy = env_out;
        d->wip_event.f0 = d->current_f0;
        d->wip_event.delta_f0 = 0.0f;
        d->wip_event.duration_s = 0;

        // Multi-feature values
        d->wip_event.spectral_flux = d->current_spectral_flux;
        d->wip_event.high_freq_energy = d->current_high_freq_energy;
        d->wip_event.mfcc_delta = d->current_mfcc_delta;
        d->wip_event.wavelet_score = d->current_wavelet_score;
        d->wip_event.fusion_score = d->current_fusion_score;
        d->wip_event.onset_type = d->current_onset_type;

        d->max_peak_rate_in_syllable = peak_rate;
        d->max_fusion_score_in_syllable = d->current_fusion_score;
        d->energy_accum = env_out;
        d->onset_timestamp = d->total_samples;
        d->peak_sample_offset = 0;

        // Reset F0 rise tracking for next event
        d->min_f0_since_peak = d->smoothed_f0;
        d->f0_has_risen = 0;
      }
    } else if (d->state == STATE_ONSET_RISING) {
      d->state_timer++;
      d->energy_accum += env_out;

      // Track Max values
      if (peak_rate > d->max_peak_rate_in_syllable) {
        d->max_peak_rate_in_syllable = peak_rate;
        d->wip_event.peak_rate = peak_rate;
        d->peak_sample_offset = d->state_timer;
      }
      if (d->current_fusion_score > d->max_fusion_score_in_syllable) {
        d->max_fusion_score_in_syllable = d->current_fusion_score;
        d->wip_event.fusion_score = d->current_fusion_score;
      }

      // Update multi-feature values to max
      if (d->current_spectral_flux > d->wip_event.spectral_flux) {
        d->wip_event.spectral_flux = d->current_spectral_flux;
      }
      if (d->current_high_freq_energy > d->wip_event.high_freq_energy) {
        d->wip_event.high_freq_energy = d->current_high_freq_energy;
      }
      if (d->current_mfcc_delta > d->wip_event.mfcc_delta) {
        d->wip_event.mfcc_delta = d->current_mfcc_delta;
      }
      if (d->current_wavelet_score > d->wip_event.wavelet_score) {
        d->wip_event.wavelet_score = d->current_wavelet_score;
      }

      // Transition to NUCLEUS when values start dropping
      int pr_dropping = (peak_rate < d->max_peak_rate_in_syllable * 0.5f);
      int fusion_dropping =
          (d->current_fusion_score < d->max_fusion_score_in_syllable * 0.6f);

      // IMPROVED: Time limit for ONSET_RISING (50ms max)
      int time_limit_reached = (d->state_timer > d->max_onset_rising_samples);

      if (pr_dropping || fusion_dropping || time_limit_reached) {
        d->state = STATE_NUCLEUS;
        float rise_time_s =
            (float)(d->peak_sample_offset + 1) / d->config.sample_rate;
        d->wip_event.pr_slope =
            d->max_peak_rate_in_syllable / (rise_time_s + 0.0001f);
      }

      // Handle unexpected voicing loss for voiced onsets
      if (!d->is_voiced && d->current_onset_type == ONSET_TYPE_VOICED) {
        d->state = STATE_COOLDOWN;
      }
    } else if (d->state == STATE_NUCLEUS) {
      d->state_timer++;
      d->energy_accum += env_out;

      // End of Nucleus Condition
      // REALTIME FIX: Use fusion-based energy comparison in realtime mode
      // since peak_rate may be near zero
      int energy_low;
      if (d->config.realtime_mode) {
        // In realtime mode, check if current energy dropped significantly
        // from the peak energy during this syllable
        float peak_energy = d->wip_event.energy > 0
                                ? d->wip_event.energy
                                : d->max_fusion_score_in_syllable;
        energy_low = (d->current_energy < peak_energy * 0.2f);
      } else {
        energy_low = (env_out < d->wip_event.peak_rate * 0.1f);
      }

      int voicing_lost =
          (!d->is_voiced && d->current_onset_type == ONSET_TYPE_VOICED);
      int fusion_low = (d->current_fusion_score < fusion_threshold_off);

      // REALTIME FIX: Add time-based exit to prevent infinite nucleus state
      // Max nucleus duration: 100ms (typical syllable peak is 50-150ms)
      int max_nucleus_samples = (int)(0.1f * d->config.sample_rate);
      int nucleus_timeout = (d->state_timer > max_nucleus_samples);

      if (energy_low || voicing_lost || fusion_low || nucleus_timeout) {
        d->state = STATE_COOLDOWN;

        // Finalize Event
        d->wip_event.duration_s =
            (float)(d->total_samples - d->onset_timestamp) /
            d->config.sample_rate;
        d->wip_event.energy = d->energy_accum;
        d->wip_event.f0 = d->current_f0;

        // Push to Ring Buffer
        if (d->buf_count < PROMINENCE_BUFFER_SIZE) {
          d->event_buffer[d->buf_write_idx].event = d->wip_event;
          d->event_buffer[d->buf_write_idx].is_ready = 1;
          d->buf_write_idx = (d->buf_write_idx + 1) % PROMINENCE_BUFFER_SIZE;
          d->buf_count++;
        } else {
          d->event_buffer[d->buf_write_idx].event = d->wip_event;
          d->event_buffer[d->buf_write_idx].is_ready = 1;
          d->buf_write_idx = (d->buf_write_idx + 1) % PROMINENCE_BUFFER_SIZE;
          d->buf_read_idx = (d->buf_read_idx + 1) % PROMINENCE_BUFFER_SIZE;
        }

        // Update last event time for F0 bypass calculation
        d->last_event_samples = d->total_samples;
      }
    } else if (d->state == STATE_COOLDOWN) {
      d->state_timer++;
      int min_dist_samples = (int)(d->config.min_syllable_dist_ms * 0.001f *
                                   d->config.sample_rate);
      if (d->state_timer > min_dist_samples) {
        d->state = STATE_IDLE;
      }
    }

    // 6. Delayed Event Emission
    // REALTIME FIX: In realtime mode, emit events immediately (no context
    // delay)
    int context_needed = d->config.realtime_mode ? 0 : d->config.context_size;

    while (d->buf_count > context_needed && events_written < max_events) {
      SyllableEvent *evt = &d->event_buffer[d->buf_read_idx].event;

      calculate_delta_f0(d, d->buf_read_idx);

      float score = calculate_prominence(d, d->buf_read_idx);
      evt->prominence_score = score;

      // 2-tier threshold: primary (1.0+) and secondary (0.7+) accents
      // Using lower thresholds to capture secondary accents like "nite" in
      // "definitely"
      evt->is_accented = (score > 0.9f); // Lower threshold for better recall

      events_out[events_written++] = *evt;

      d->event_buffer[d->buf_read_idx].is_ready = 0;
      d->buf_read_idx = (d->buf_read_idx + 1) % PROMINENCE_BUFFER_SIZE;
      d->buf_count--;
    }
  }

  return events_written;
}

int syllable_flush(SyllableDetector *d, SyllableEvent *events_out,
                   int max_events) {
  int events_written = 0;

  while (d->buf_count > 0 && events_written < max_events) {
    SyllableEvent *evt = &d->event_buffer[d->buf_read_idx].event;

    calculate_delta_f0(d, d->buf_read_idx);

    float score = calculate_prominence(d, d->buf_read_idx);
    evt->prominence_score = score;
    evt->is_accented = (score > 1.2f);

    events_out[events_written++] = *evt;

    d->event_buffer[d->buf_read_idx].is_ready = 0;
    d->buf_read_idx = (d->buf_read_idx + 1) % PROMINENCE_BUFFER_SIZE;
    d->buf_count--;
  }

  return events_written;
}

// --- Real-Time Mode API ---

/**
 * @brief Enable or disable real-time mode
 * @param d Detector instance
 * @param enable 1 to enable, 0 to disable
 */
void syllable_set_realtime_mode(SyllableDetector *d, int enable) {
  if (!d)
    return;
  d->config.realtime_mode = enable ? 1 : 0;
  if (enable) {
    // Auto-start calibration when enabling RT mode
    memset(&d->rt_cal, 0, sizeof(d->rt_cal));
    d->rt_cal.is_calibrating = 1;
    d->rt_cal.target_samples = (int)(d->config.calibration_duration_ms *
                                     0.001f * d->config.sample_rate);
  }
}

/**
 * @brief Reset calibration and restart data collection
 * @param d Detector instance
 * @note If realtime_mode is disabled, this function enables it automatically
 */
void syllable_recalibrate(SyllableDetector *d) {
  if (!d)
    return;
  // Auto-enable RT mode if not already enabled
  if (!d->config.realtime_mode) {
    d->config.realtime_mode = 1;
  }
  memset(&d->rt_cal, 0, sizeof(d->rt_cal));
  d->rt_cal.is_calibrating = 1;
  d->rt_cal.target_samples =
      (int)(d->config.calibration_duration_ms * 0.001f * d->config.sample_rate);
}

/**
 * @brief Check if detector is currently in calibration phase
 * @param d Detector instance
 * @return 1 if calibrating, 0 otherwise
 */
int syllable_is_calibrating(SyllableDetector *d) {
  return d ? d->rt_cal.is_calibrating : 0;
}

/**
 * @brief Set SNR threshold for real-time mode detection
 * @param d Detector instance
 * @param snr_db SNR threshold in dB (default: 6.0, lower = more sensitive)
 */
void syllable_set_snr_threshold(SyllableDetector *d, float snr_db) {
  if (!d)
    return;
  d->config.snr_threshold_db = snr_db;
  // Update gamma immediately if already calibrated
  if (!d->rt_cal.is_calibrating && d->config.realtime_mode) {
    d->rt_cal.gamma = powf(10.0f, snr_db / 10.0f);
  }
}
