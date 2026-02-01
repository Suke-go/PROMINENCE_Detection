# Prominence Detection System

音節とプロミネンス（韻律的卓立）を検出するリアルタイム音声処理システム。

## 🎯 プロジェクト構成

```
syllabledetection/
├── core/           # 根幹技術 - C言語DSPエンジン
├── models/piw/     # 新SSL - Physically-Inductive Wav2Vec
├── demos/web/      # Webデモ - リアルタイムProminence検出
├── papers/         # 論文・研究資料
└── docs/           # ドキュメント
```

---

## 🔧 Core Engine (`core/`)

リアルタイム音声処理のためのC言語ライブラリ **libsyllable**。

### 特徴
- **Multi-Feature Fusion**: ZFF, Spectral Flux, MFCC, Wavelet等を統合
- **WebAssembly対応**: ブラウザで動作
- **低レイテンシ**: 20ms以下

### ビルド
```bash
cd core
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

📖 [詳細ドキュメント](core/README.md)

---

## 🧠 PIW Model (`models/piw/`)

**Physically-Inductive Wav2Vec** - 音声物理学に基づく自己教師あり学習モデル。

### 特徴
- **Source-Filter Theory**: 声帯振動と声道フィルタを Inductive Bias として導入
- **Data Efficiency**: 10時間データで従来の100時間相当の性能
- **Interpretability**: 学習されたF0/Envelope τ を解析可能

### 使用方法
```bash
cd models/piw
pip install -r requirements.txt

# 事前学習
python scripts/pretrain.py --config configs/pretrain_10h.yaml

# Fine-tuning
python scripts/finetune.py --config configs/finetune_prominence.yaml
```

📖 [詳細ドキュメント](models/piw/README.md)

---

## 🎮 Web Demo (`demos/web/`)

ブラウザでリアルタイムにProminenceを検出するデモ。

### 起動
```bash
cd demos/web
python -m http.server 8000
# ブラウザで http://localhost:8000 を開く
```

📖 [詳細ドキュメント](demos/web/README.md)

---

## 📄 Papers (`papers/`)

- **visceral_resonance/**: Augmented Humans 2026 Demo Paper

---

## 🛠 Requirements

### Core Engine
- C99コンパイラ (GCC/Clang/MSVC)
- CMake 3.10+
- Emscripten SDK (Wasmビルド用)

### PIW Model
- Python 3.10+
- PyTorch 2.0+
- CUDA 12.x (GPU使用時)

---

## 📝 License

MIT License
