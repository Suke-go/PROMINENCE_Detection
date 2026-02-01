# Physically-Inductive Wav2Vec (PIW)

音声の物理的生成モデル（Source-Filter Theory）に基づく Inductive Bias を Deep Learning に組み込んだ、Data-Efficient な Prominence Detection システム。

## Architecture

```
Raw Audio → [Inductive Front-end] → [Transformer 6L] → [Head]
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 Source          Filter          Rhythm
 (Harmonic       (Causal         (AM
  SincNet)        Conv)           Filterbank)
```

## Quick Start

### 1. Setup

```bash
# 仮想環境
python -m venv .venv

# Windows
.venv\Scripts\activate

# 依存関係
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# LibriSpeech (事前学習用)
# 環境変数を設定するか、--data_root で指定
export LIBRISPEECH_ROOT=/path/to/librispeech

# Helsinki Prosody (Fine-tuning用)
export HELSINKI_PROSODY_ROOT=/path/to/helsinki
```

### 3. Pretraining

```bash
# 1時間データで事前学習 (Data Efficiency実験)
python scripts/pretrain.py --config configs/pretrain_1h.yaml

# 10時間データで事前学習
python scripts/pretrain.py --config configs/pretrain_10h.yaml

# デバッグモード (100ステップのみ)
python scripts/pretrain.py --config configs/pretrain_1h.yaml --debug
```

### 4. Fine-tuning

```bash
python scripts/finetune.py \
    --config configs/finetune_prominence.yaml \
    --pretrained outputs/pretrain/final_model.pt
```

### 5. Evaluation

```bash
# テストセット評価
python scripts/evaluate.py \
    --model outputs/finetune/best_model.pt \
    --data_root /path/to/helsinki

# Ablation Study
python scripts/ablation.py \
    --model outputs/finetune/best_model.pt \
    --data_root /path/to/helsinki
```

## Project Structure

```
piw/
├── configs/
│   ├── base.yaml              # 基本設定
│   ├── pretrain_1h.yaml       # 1h事前学習
│   ├── pretrain_10h.yaml      # 10h事前学習
│   └── finetune_prominence.yaml
│
├── src/
│   ├── frontend/              # Inductive Front-end
│   │   ├── harmonic_sincnet.py  # Source Branch
│   │   ├── causal_filterbank.py # Filter Branch
│   │   ├── am_filterbank.py     # Rhythm Branch
│   │   ├── causal_envelope.py   # IIR Envelope
│   │   └── frontend.py          # 統合モジュール
│   │
│   ├── encoder/               # Transformer
│   │   ├── transformer.py
│   │   └── position_encoding.py
│   │
│   ├── heads/                 # Task Heads
│   │   ├── masked_prediction.py
│   │   └── prominence.py
│   │
│   ├── model.py               # PIWModel
│   └── dataset.py             # データローダ
│
├── scripts/
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── ablation.py
│
└── tests/
    ├── test_harmonic_sincnet.py
    ├── test_causal_envelope.py
    ├── test_frontend.py
    └── test_model.py
```

## Hardware Requirements

- **GPU**: NVIDIA Blackwell (RTX 5090) または 48GB+ VRAM
- **CUDA**: 12.x
- **RAM**: 32GB+

## Expected Results

| Data | Pretrain Steps | F1 (Target) |
|------|----------------|-------------|
| 1h   | 10k            | ≥ 0.80      |
| 10h  | 50k            | ≥ 0.85      |

## License

MIT
