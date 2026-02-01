using System;
using System.Runtime.InteropServices;

namespace SyllableDetection
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SyllableConfig
    {
        public int sample_rate;

        // ZFF Config
        public float zff_trend_window_ms;

        // PeakRate Config
        public float peak_rate_band_min;
        public float peak_rate_band_max;

        // Detection Logic
        public float min_syllable_dist_ms;
        public float threshold_peak_rate;
        public float adaptive_peak_rate_k;
        public float adaptive_peak_rate_tau_ms;
        public float voiced_hold_ms;

        // Hysteresis
        public float hysteresis_on_factor;
        public float hysteresis_off_factor;

        // Prominence Context
        public int context_size;

        // User Memory (function pointers)
        public IntPtr user_malloc;
        public IntPtr user_free;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SyllableEvent
    {
        public ulong timestamp_samples;
        public double time_seconds;

        // Feature Values
        public float peak_rate;
        public float pr_slope;
        public float energy;
        public float f0;
        public float delta_f0;
        public float duration_s;

        // Prominence / Accent
        public float prominence_score;
        public int is_accented;
    }

    public class SyllableDetector : IDisposable
    {
        private IntPtr _handle;

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr syllable_create(ref SyllableConfig config);

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern void syllable_reset(IntPtr detector);

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern int syllable_process(IntPtr detector, float[] input, int num_samples, [Out] SyllableEvent[] events_out, int max_events);

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern int syllable_flush(IntPtr detector, [Out] SyllableEvent[] events_out, int max_events);

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern void syllable_destroy(IntPtr detector);

        [DllImport("syllable", CallingConvention = CallingConvention.Cdecl)]
        private static extern SyllableConfig syllable_default_config(int sample_rate);


        public SyllableDetector(int sampleRate)
        {
            var config = syllable_default_config(sampleRate);
            _handle = syllable_create(ref config);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Failed to create SyllableDetector");
            }
        }

        public SyllableDetector(SyllableConfig config)
        {
            _handle = syllable_create(ref config);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Failed to create SyllableDetector");
            }
        }
        
        // Helper to get default config for modification
        public static SyllableConfig GetDefaultConfig(int sampleRate)
        {
             return syllable_default_config(sampleRate);
        }

        public void Reset()
        {
            if (_handle != IntPtr.Zero)
            {
                syllable_reset(_handle);
            }
        }

        public int Process(float[] input, SyllableEvent[] eventsOut)
        {
            if (_handle == IntPtr.Zero) return 0;
            return syllable_process(_handle, input, input.Length, eventsOut, eventsOut.Length);
        }

        public int Flush(SyllableEvent[] eventsOut)
        {
            if (_handle == IntPtr.Zero) return 0;
            return syllable_flush(_handle, eventsOut, eventsOut.Length);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                syllable_destroy(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }
}
