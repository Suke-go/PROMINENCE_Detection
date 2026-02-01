import ctypes
import os
import sys

# Define structures matching C implementation
class SyllableConfig(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_int),
        ("zff_trend_window_ms", ctypes.c_float),
        ("peak_rate_band_min", ctypes.c_float),
        ("peak_rate_band_max", ctypes.c_float),
        ("min_syllable_dist_ms", ctypes.c_float),
        ("threshold_peak_rate", ctypes.c_float),
        ("adaptive_peak_rate_k", ctypes.c_float),
        ("adaptive_peak_rate_tau_ms", ctypes.c_float),
        ("voiced_hold_ms", ctypes.c_float),
        ("hysteresis_on_factor", ctypes.c_float),
        ("hysteresis_off_factor", ctypes.c_float),
        ("context_size", ctypes.c_int),
        ("user_malloc", ctypes.c_void_p),
        ("user_free", ctypes.c_void_p),
    ]

# Event structure
class SyllableEvent(ctypes.Structure):
    _fields_ = [
        ("timestamp_samples", ctypes.c_uint64),
        ("time_seconds", ctypes.c_double),
        ("peak_rate", ctypes.c_float),
        ("pr_slope", ctypes.c_float),
        ("energy", ctypes.c_float),
        ("f0", ctypes.c_float),
        ("delta_f0", ctypes.c_float),
        ("duration_s", ctypes.c_float),
        ("prominence_score", ctypes.c_float),
        ("is_accented", ctypes.c_int),
    ]

class SyllableDetector:
    def __init__(self, lib_path, sample_rate=44100):
        # Load Library
        if sys.platform == "win32":
            self.lib = ctypes.CDLL(lib_path)
        else:
            self.lib = ctypes.CDLL(lib_path)

        # Function signatures
        self.lib.syllable_default_config.argtypes = [ctypes.c_int]
        self.lib.syllable_default_config.restype = SyllableConfig

        self.lib.syllable_create.argtypes = [ctypes.POINTER(SyllableConfig)]
        self.lib.syllable_create.restype = ctypes.c_void_p

        self.lib.syllable_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(SyllableEvent),
            ctypes.c_int
        ]
        self.lib.syllable_process.restype = ctypes.c_int

        self.lib.syllable_flush.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(SyllableEvent),
            ctypes.c_int
        ]
        self.lib.syllable_flush.restype = ctypes.c_int

        self.lib.syllable_destroy.argtypes = [ctypes.c_void_p]
        self.lib.syllable_reset.argtypes = [ctypes.c_void_p]

        # Init
        self.config = self.lib.syllable_default_config(sample_rate)
        self.handle = self.lib.syllable_create(ctypes.byref(self.config))
        if not self.handle:
            raise RuntimeError("Failed to create SyllableDetector instance")

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.syllable_destroy(self.handle)

    def process_block(self, float_samples):
        # Handles list or numpy array (must be strictly cast to float array)
        num_samples = len(float_samples)
        ArrayType = ctypes.c_float * num_samples
        
        # If it's already a ctypes array, use it directly, otherwise convert
        if isinstance(float_samples, ArrayType):
            input_ptr = float_samples
        else:
            # This is slow for large buffers without numpy
            # Recommended to use: (ctypes.c_float * len(data))(*data) outside
            input_ptr = ArrayType(*float_samples)

        max_events = 64
        EventType = SyllableEvent * max_events
        events_out = EventType()

        count = self.lib.syllable_process(self.handle, input_ptr, num_samples, events_out, max_events)
        
        results = []
        for i in range(count):
            e = events_out[i]
            # Copy to python dict or object to detach from C memory if needed
            results.append({
                'time': e.time_seconds,
                'peak_rate': e.peak_rate,
                'score': e.prominence_score,
                'is_accented': bool(e.is_accented),
                'f0': e.f0,
                'slope': e.pr_slope
            })
        return results

    def flush(self):
        max_events = 64
        EventType = SyllableEvent * max_events
        events_out = EventType()
        count = self.lib.syllable_flush(self.handle, events_out, max_events)
        
        results = []
        for i in range(count):
            e = events_out[i]
            results.append({
                'time': e.time_seconds,
                'peak_rate': e.peak_rate,
                'score': e.prominence_score,
                'is_accented': bool(e.is_accented),
                'f0': e.f0,
                'slope': e.pr_slope
            })
        return results

if __name__ == "__main__":
    # Simple test execution
    print("SyllableDetector Python Wrapper")
    # Usage: detector = SyllableDetector("path/to/syllable.dll", 44100)
