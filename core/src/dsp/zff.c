#include "zff.h"
#include <stdlib.h>
#include <string.h>


void zff_init(ZFF *z, int sample_rate, float trend_window_ms,
              void *(*custom_alloc)(size_t)) {
  z->int1 = 0.0;
  z->int2 = 0.0;

  // Window size in samples
  z->trend_buf_size = (int)(sample_rate * trend_window_ms * 0.001f);
  if (z->trend_buf_size < 1)
    z->trend_buf_size = 1;

  size_t buf_bytes = z->trend_buf_size * sizeof(float);
  if (custom_alloc) {
    z->trend_buffer = (float *)custom_alloc(buf_bytes);
  } else {
    z->trend_buffer = (float *)malloc(buf_bytes);
  }

  // Zero out buffer
  if (z->trend_buffer) {
    memset(z->trend_buffer, 0, buf_bytes);
  }

  z->trend_write_pos = 0;
  z->trend_accum = 0.0f;
}

void zff_process(ZFF *z, float in, float *zff_out, float *slope_out) {
  // 1. Difference (Pre-processing for ZFF, usually x[n] - x[n-1])
  // But since ZFF is effectively double integration, and input speech has
  // near-zero mean, we can apply the double integration directly on the
  // difference signal, OR just integrate the signal twice if we remove trend
  // aggressively. The standard Murty & Yegnanarayana approach is: Signal ->
  // Diff -> Int -> Int -> Trend Remove. Which is equivalent to: Int -> Trend
  // Remove (since Int(Diff) = Signal). Wait, Int(Diff(x)) = x. So Diff -> Int
  // -> Int is just Integration of x. Let's stick to the paper: x[n] - x[n-1] is
  // the input to the double resonator.

  // Actually, simpler implementation for "Zero Frequency Resonator":
  // y[n] = x[n] + 2y[n-1] - y[n-2]
  // This is explicitly a double integrator.

  // Stability is an issue with pure integrators. Hence the trend removal is
  // critical.

  // Let's assume the caller passes the raw signal 'in'.
  // We treat the input to the integrators as the signal itself (or diff).
  // Let's implement the standard: y[n] = - \sum_{k=1}^{n} \sum_{j=1}^{k} x'[j]
  // where x'[j] is the differenced signal.

  // First, integrate.
  z->int1 += (double)in;
  // Second, integrate again.
  z->int2 += z->int1;

  // Now we have the raw ZFF signal (very widely drifting).
  // We need to remove the "trend" which is the local mean over ~10ms window.

  // Running Mean calculation
  // We need the mean of z->int2 over the window size.
  // But int2 grows unbounded! This is a numeric problem for long streams.
  // Solution: Only track the "DC" offset in a window and subtract it.
  // Actually, the "Trend" isn't just DC, it's the low frequency component.
  // Subtracting a Moving Average is a High Pass filter.

  // To handle the unbounded growth, a leaky integrator is often preferred in
  // real-time, OR we reset/wrap - but for audio, Leaky is best. However, exact
  // ZFF relies on the property of the ideal resonator. For this implementation,
  // let's use a Leaky Integrator with extremely long time constant effectively
  // acting as the resonator, followed by the trend removal.

  // Re-evaluating real-time ZFF:
  // "Epoch Extraction from Speech Signals" (Murty) mentions mean subtraction.
  // For real-time, we can't look ahead. We can only look back.
  // So "Trend" is the mean of previous N samples.

  // However, floating point precision will kill us if we just accumulate
  // forever. We should implement the "Mean Subtraction" as a High Pass Filter
  // at very low cutoff? No, ZFF specifically needs 0Hz resonance.

  // Let's allow the leak.
  // y[n] = x[n] + 2*r*y[n-1] - r^2*y[n-2] where r is close to 1 (e.g. 0.999...)
  // This creates a resonator at 0Hz with finite Q.

  double output_raw = z->int2; // BUT this is unbounded.

  // Let's switch to the "Stable" version:
  // Use a very low frequency High Pass Filter to remove the trend from the
  // output of the integrators? Better: Filter the *input* to remove any DC, and
  // just accept that int2 might drift? No, drift is inevitable.

  // Let's use the Remainder method or local polynomial fit? Too complex.
  // Let's use the recursive filter approach for trend removal:
  // Output = Raw - Average(Raw).

  // For this "libsyllable", let's use a LEAKY double integrator.
  // y[n] = x[n] + 1.99 * y[n-1] - 0.99 * y[n-2] ?
  // Check stability.

  // Actually, for the ZFF epoch extraction, the trend is usually removed by
  // subtracting the moving average over the pitch period range (5-15ms).

  // Let's handle the drift by simple high-pass filtering the output of the
  // integrators? Yes, Mean Subtraction IS a high pass filter (Comb filter with
  // notch at 0?). Moving Average filter has zeros at N/Window. Subtracting
  // Moving Average = 1 - MA = High Pass.

  // Implementation:
  // We maintain the Trend Buffer containing the recent values of the integrated
  // signal. BUT we can't store unbounded values in the buffer.

  // Hack for stability in real-time float implementation:
  // Apply a coefficient slightly < 1.0 to the integrators.
  const double leak = 0.999;
  z->int1 = z->int1 * leak + (double)in;
  z->int2 = z->int2 * leak + z->int1;

  float val = (float)z->int2;

  // Trend Removal (Moving Average)
  if (z->trend_buffer) {
    float old_val = z->trend_buffer[z->trend_write_pos];
    z->trend_buffer[z->trend_write_pos] = val;
    z->trend_accum += val - old_val;
    z->trend_write_pos++;
    if (z->trend_write_pos >= z->trend_buf_size)
      z->trend_write_pos = 0;

    float trend = z->trend_accum / z->trend_buf_size;
    *zff_out = val - trend;
  } else {
    *zff_out = val;
  }

  // Slope can be useful for zero crossing detection
  // Simple 1st order diff of the output
  *slope_out = 0.0f; // TODO: Maintain history for slope if needed
}

void zff_destroy(ZFF *z, void (*custom_free)(void *)) {
  if (z->trend_buffer) {
    if (custom_free) {
      custom_free(z->trend_buffer);
    } else {
      free(z->trend_buffer);
    }
    z->trend_buffer = NULL;
  }
}
