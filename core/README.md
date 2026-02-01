# libsyllable - Core DSP Engine

リアルタイム音節・プロミネンス検出のためのC言語ライブラリ。

## アーキテクチャ

```
Audio Input → [AGC] → [Feature Extraction] → [Fusion] → [Peak Detection] → Events
                            │
            ┌───────────────┼───────────────┐───────────────┐
            ▼               ▼               ▼               ▼
         Spectral        MFCC           Wavelet          ZFF
          Flux          Delta          Transform        (F0)
```

## DSPモジュール

| モジュール | ファイル | 役割 |
|-----------|----------|------|
| AGC | `src/dsp/agc.c` | 自動ゲイン制御 |
| Spectral Flux | `src/dsp/spectral_flux.c` | スペクトル変動量 |
| MFCC | `src/dsp/mfcc.c` | ケプストラム特徴量 |
| Wavelet | `src/dsp/wavelet.c` | トランジェント検出 |
| ZFF | `src/dsp/zff.c` | 有声/無声判定、F0 |
| High-Freq Energy | `src/dsp/high_freq_energy.c` | 子音・破裂音検出 |

## ビルド

### ネイティブ
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### WebAssembly
```bash
# Emscripten SDK セットアップ後
emcmake cmake ..
emmake make
```

## 使用例

```c
#include "syllable_detector.h"

SyllableConfig config = syllable_default_config(16000);
SyllableDetector *detector = syllable_create(&config);

float audio[1024];
SyllableEvent events[64];
int count = syllable_process(detector, audio, 1024, events, 64);

syllable_destroy(detector);
```

## API

| 関数 | 説明 |
|------|------|
| `syllable_create()` | 検出器を作成 |
| `syllable_process()` | 音声を処理 |
| `syllable_destroy()` | 検出器を破棄 |
| `syllable_set_realtime_mode()` | RTモード切替 |
| `syllable_recalibrate()` | 再キャリブレーション |

## 依存関係

- [KissFFT](https://github.com/mborgerding/kissfft) (extern/kissfft)
