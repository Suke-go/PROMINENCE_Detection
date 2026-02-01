#ifndef SYLLABLE_DETECTOR_H
#define SYLLABLE_DETECTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef EXPORT_DLL
#define SYLLABLE_API __declspec(dllexport)
#else
#define SYLLABLE_API __declspec(dllimport)
#endif
#else
#define SYLLABLE_API
#endif

// --- Configuration ---

typedef struct {
  int sample_rate;

  // ZFF Config
  float zff_trend_window_ms; // Window for removing low-frequency trend
                             // (default: 10.0)

  // PeakRate Config
  float peak_rate_band_min; // Bandpass min Hz (default: 500.0)
  float peak_rate_band_max; // Bandpass max Hz (default: 3200.0)

  // Detection Logic
  float min_syllable_dist_ms; // Minimum distance between syllables (default:
                              // 100.0)
  float threshold_peak_rate;  // Absolute floor for peakRate threshold (default:
                              // 0.0003)
  float adaptive_peak_rate_k; // Adaptive threshold = mean + k*std (0 disables,
                              // default: 4.0)
  float adaptive_peak_rate_tau_ms; // Time constant for adaptive stats (default:
                                   // 500.0)
  float voiced_hold_ms;            // Voicing hold window (default: 30.0)

  // Hysteresis (to prevent chattering)
  float hysteresis_on_factor;  // Multiplier for ON threshold (default: 1.3)
  float hysteresis_off_factor; // Multiplier for OFF threshold (default: 0.7)

  // Prominence Context
  int context_size; // Number of syllables to look ahead/behind (default: 2)

  // --- Multi-Feature Detection (NEW) ---

  // Feature enables (set to 0 to disable)
  int enable_spectral_flux;    // Enable Spectral Flux detection (default: 1)
  int enable_high_freq_energy; // Enable high-freq energy tracking (default: 1)
  int enable_mfcc_delta;       // Enable MFCC delta detection (default: 1)
  int enable_wavelet; // Enable Wavelet Transform detection (default: 1)

  // FFT parameters for spectral features
  float fft_size_ms; // FFT window size in ms (default: 32.0)
  float hop_size_ms; // Hop size in ms (default: 16.0)

  // High-frequency energy config
  float high_freq_cutoff_hz; // High-pass cutoff for HFE (default: 2000.0)

  // Feature Fusion weights (should sum to ~1.0)
  float weight_peak_rate;     // Weight for PeakRate (default: 0.25)
  float weight_spectral_flux; // Weight for Spectral Flux (default: 0.20)
  float weight_high_freq;     // Weight for High-Freq Energy (default: 0.15)
  float weight_mfcc_delta;    // Weight for MFCC Delta (default: 0.10)
  float weight_wavelet;       // Weight for Wavelet Transform (default: 0.20)
  float weight_voiced_bonus;  // Weight for voiced bonus (default: 0.10)

  // Fusion blend ratio: fusion = alpha * max + (1-alpha) * weighted_avg
  float fusion_blend_alpha; // Blend ratio for max/avg (default: 0.6)

  // Unvoiced onset detection
  float
      unvoiced_onset_threshold; // Threshold for unvoiced onsets (default: 0.5)
  int allow_unvoiced_onsets;    // Allow onsets without voicing (default: 1)

  // Robustness defaults
  int enable_agc; // Enable Automatic Gain Control (default: 1)

  // --- Real-Time Mode (NEW) ---
  int realtime_mode; // 0=offline (adaptive), 1=realtime (default: 0)
  float calibration_duration_ms; // Calibration duration in ms (default: 2000.0)
  float snr_threshold_db;        // SNR threshold in dB (default: 6.0)

  // --- Hierarchical Prominence (NEW) ---
  // Duration as continuous feature (Fry 1955)
  float duration_weight;    // Weight for duration modulation (default: 0.1)
  float duration_center_ms; // Center point for sigmoid (default: 80.0)
  float duration_scale_ms;  // Scale for sigmoid steepness (default: 30.0)

  // Pause-based phrase detection
  float pause_threshold_ms;   // Silence duration to detect phrase boundary
                              // (default: 200.0)
  float phrase_initial_boost; // Boost for post-pause syllables (default: 0.15)

  // User Memory (Optional, set to NULL to use malloc/free)
  void *(*user_malloc)(size_t);
  void (*user_free)(void *);
} SyllableConfig;

// --- Onset Type ---
typedef enum {
  ONSET_TYPE_VOICED = 0,   // Traditional voiced onset
  ONSET_TYPE_UNVOICED = 1, // Unvoiced consonant (plosive, fricative)
  ONSET_TYPE_MIXED = 2     // Mixed (e.g., voiced fricative)
} SyllableOnsetType;

// --- Output Event ---

typedef struct {
  uint64_t timestamp_samples; // Sample index of the onset
  double time_seconds;        // Time in seconds

  // Feature Values (Legacy)
  float peak_rate;  // Max slope at onset
  float pr_slope;   // PeakRate rise slope (peak_rate / rise_time_s)
  float energy;     // Integrated energy
  float f0;         // Fundamental frequency (ZFF derived)
  float delta_f0;   // F0 difference from context median
  float duration_s; // Estimated duration

  // Multi-Feature Values (NEW)
  float spectral_flux;          // Spectral Flux value at onset
  float high_freq_energy;       // High-frequency energy at onset
  float mfcc_delta;             // MFCC change magnitude at onset
  float wavelet_score;          // Wavelet transient score at onset
  float fusion_score;           // Combined detection score (0-1+)
  SyllableOnsetType onset_type; // Type of onset detected

  // Prominence / Accent
  float prominence_score; // 0.0 to 1.0 (or higher), relative to context
  int is_accented;        // 1 if accented, 0 otherwise
} SyllableEvent;

// --- Opaque Handle ---
typedef struct SyllableDetector SyllableDetector;

// --- API Functions ---

// Create a new detector instance
SYLLABLE_API SyllableDetector *syllable_create(const SyllableConfig *config);

// Reset internal state (e.g. for new file)
SYLLABLE_API void syllable_reset(SyllableDetector *detector);

// Process a block of audio samples (mono, float)
// Returns the number of events detected in this block (that are now "ready"
// after context delay) Populates 'events_out' up to 'max_events' capacity.
SYLLABLE_API int syllable_process(SyllableDetector *detector,
                                  const float *input, int num_samples,
                                  SyllableEvent *events_out, int max_events);

// Flush any remaining events in the buffer (e.g. at end of file)
SYLLABLE_API int syllable_flush(SyllableDetector *detector,
                                SyllableEvent *events_out, int max_events);

// Destroy the instance
SYLLABLE_API void syllable_destroy(SyllableDetector *detector);

// Helper to get default config
SYLLABLE_API SyllableConfig syllable_default_config(int sample_rate);

// --- Real-Time Mode API (NEW) ---

/**
 * @brief Enable or disable real-time mode
 * @param detector Detector instance
 * @param enable 1 to enable, 0 to disable
 */
SYLLABLE_API void syllable_set_realtime_mode(SyllableDetector *detector,
                                             int enable);

/**
 * @brief Reset calibration and start real-time mode
 * @param detector Detector instance
 * @note If realtime_mode is disabled, this function enables it automatically
 */
SYLLABLE_API void syllable_recalibrate(SyllableDetector *detector);

/**
 * @brief Check if detector is currently calibrating
 * @param detector Detector instance
 * @return 1 if calibrating, 0 otherwise
 */
SYLLABLE_API int syllable_is_calibrating(SyllableDetector *detector);

/**
 * @brief Set SNR threshold for real-time mode detection
 * @param detector Detector instance
 * @param snr_db SNR threshold in dB (default: 6.0, lower = more sensitive)
 * @note Lower values make detection more sensitive, higher values more strict
 */
SYLLABLE_API void syllable_set_snr_threshold(SyllableDetector *detector,
                                             float snr_db);

#ifdef __cplusplus
}
#endif

#endif // SYLLABLE_DETECTOR_H
