cmake_minimum_required(VERSION 3.10)
project(libsyllable VERSION 0.1.0 LANGUAGES C)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_EXAMPLES "Build example applications" OFF)
option(BUILD_TESTS "Build unit tests" OFF)

# Include directories
include_directories(include)
include_directories(src)        # For DSP headers
include_directories(extern)     # For kissfft

# Source files
set(SOURCES
    src/syllable_detector.c
    src/dsp/biquad.c
    src/dsp/envelope.c
    src/dsp/agc.c
    src/dsp/zff.c
    src/dsp/spectral_flux.c
    src/dsp/high_freq_energy.c
    src/dsp/mfcc.c
    src/dsp/wavelet.c
    src/peak_detector.c
    extern/kissfft/kiss_fft.c
    extern/kissfft/kiss_fftr.c
)

# Library target
add_library(syllable ${SOURCES})
target_include_directories(syllable PUBLIC include)
target_include_directories(syllable PRIVATE src extern)

# Compiler warnings and SIMD optimization
if(MSVC)
    target_compile_options(syllable PRIVATE /W4 /O2)
    target_compile_definitions(syllable PRIVATE EXPORT_DLL)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # x64 has SSE2 enabled by default
    else()
        target_compile_options(syllable PRIVATE /arch:SSE2)
    endif()
else()
    target_compile_options(syllable PRIVATE -Wall -Wextra -pedantic -O3)
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-msse2" HAS_SSE2)
    check_c_compiler_flag("-mavx2" HAS_AVX2)
    if(HAS_AVX2)
        target_compile_options(syllable PRIVATE -mavx2 -mfma)
    elseif(HAS_SSE2)
        target_compile_options(syllable PRIVATE -msse2)
    endif()
endif()

# Link math library on Unix
if(UNIX)
    target_link_libraries(syllable m)
endif()

# Installation
install(TARGETS syllable DESTINATION lib)
install(FILES include/syllable_detector.h DESTINATION include)
