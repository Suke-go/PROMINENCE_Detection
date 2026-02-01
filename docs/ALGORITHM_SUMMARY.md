# Real-Time Prominence Detection Algorithm Summary

> **目的**: 指導教官への説明資料 - アルゴリズムの科学的根拠とBURNCデータセット購入の必要性

---

## 1. アルゴリズム概要

### 1.1 問題設定

リアルタイム発話フィードバックシステムにおいて、音節のプロミネンス（強調）を**20ms以下のレイテンシ**で検出する必要がある。

### 1.2 従来手法の限界

| アプローチ | 精度 | レイテンシ | 説明可能性 |
|-----------|------|-----------|-----------|
| DNN (wav2vec2) | 高 | 200-500ms | ✗ ブラックボックス |
| オフライン DSP | 中 | オフライン | ✓ 解釈可能 |
| **本手法** | **競争的** | **<20ms** | **✓ 特徴ベース** |

### 1.3 本研究のアプローチ

**6つの音響特徴量** × **幾何平均Fusion** × **オンラインキャリブレーション**

---

## 2. 特徴量設計

### 2.1 6つの特徴量

| 特徴量 | 計算方法 | 音声学的役割 | 参考文献 |
|--------|---------|-------------|---------|
| **Energy** | $\Sigma x[n]^2$ | 音量・強度 | Stevens (1998) |
| **Peak Rate** | $\max(0, e[n]-e[n-1])$ | 母音オンセット | Oganian & Chang (2019) |
| **Spectral Flux** | $\Sigma\max(0, |X_n|-|X_{n-1}|)^2$ | 音素境界 | Bello et al. (2005) |
| **High-Freq Energy** | 2-8kHz帯域エネルギー | 無声子音 | Stevens (1998) |
| **MFCC Delta** | $\Delta MFCC_1$ | 音色変化 | Rabiner (1993) |
| **Wavelet** | DWTトランジェント | 過渡検出 | Mallat (2009) |

### 2.2 特徴量選択の根拠

**Peak Rate** の重要性:
- Oganian & Chang (2019) は、人間の聴覚皮質が包絡線の立ち上がり（Peak Rate）を音節符号化の主要ランドマークとして使用することを発見
- Science Advances誌掲載の神経科学的エビデンス

**Spectral Flux** の重要性:
- 音楽情報検索（MIR）分野で標準的なオンセット検出手法
- Bello et al. (2005) IEEE Tutorial で体系化

---

## 3. 幾何平均Fusion

### 3.1 数式

各特徴量 $f_k$ について閾値 $\theta_k$ との比:
$$r_k = \frac{f_k}{\theta_k}$$

閾値を超えた ($r_k > 1$) 特徴量 $n$ 個の幾何平均:
$$G = \sqrt[n]{\prod_{k \in \mathcal{A}} r_k} = \exp\left(\frac{1}{n}\sum_{k \in \mathcal{A}} \ln r_k\right)$$

正規化スコア:
$$S = \frac{G}{2 + G} \in [0, 1]$$

### 3.2 なぜ幾何平均か？

| 平均タイプ | 1つ外れ値 | 全体一致 | ノイズ耐性 |
|-----------|----------|---------|-----------|
| 算術平均 | 高スコア | 高スコア | 低 |
| **幾何平均** | 中スコア | 高スコア | **高** |
| 最小値 | 低スコア | 高スコア | 最高 |

**幾何平均の利点**:
- 単一特徴量のノイズスパイクに鈍感
- 複数特徴量が「同時に」閾値を超えることを要求
- 音声学的に正しい（本当のプロミネンスは複数の音響手がかりを持つ）

### 3.3 重み付けの考え方

**オフラインモード**: 明示的な重み付き平均
```
Fusion = α × max(features) + (1-α) × weighted_avg(features)
```

| 特徴量 | 重み | 根拠 |
|--------|-----|------|
| Peak Rate | 0.30 | 神経科学的エビデンス [Oganian 2019] |
| Spectral Flux | 0.25 | 音素境界検出 [Bello 2005] |
| Wavelet | 0.20 | マルチスケール過渡 |
| High-Freq | 0.15 | 子音検出補助 |
| MFCC Delta | 0.10 | 音色変化補助 |

**リアルタイムモード**: 幾何平均による暗黙的等重み
- 各特徴が閾値を超えなければ寄与しない（AND的な振る舞い）
- ノイズに対してより保守的

---

## 4. オンラインキャリブレーション

### 4.1 アルゴリズム

1. 最初の2秒間で環境ノイズを収集
2. 各特徴量の平均 $\mu_k$ と標準偏差 $\sigma_k$ を計算
3. SNRベース閾値を設定:
   $$\theta_k = \mu_k + \gamma \cdot \sigma_k$$
   $$\gamma = 10^{\text{SNR}_{\text{dB}}/10}$$

### 4.2 SNR設定の意味

| SNR (dB) | γ | 意味 |
|----------|---|------|
| 6 (デフォルト) | 4.0 | ノイズフロア + 4σ |
| 3 | 2.0 | より敏感 |
| 10 | 10.0 | より鈍感 |

### 4.3 科学的根拠

- 信号検出理論 (SDT) に基づく閾値設定
- $\gamma \cdot \sigma$ は detectability index $d'$ に相当
- 心理物理学的に意味のある感度調整

---

## 5. 評価の必要性：BURNC購入要求

### 5.1 なぜBURNCが必要か？

**BURNC (Boston University Radio News Corpus)**:
- ToBI (Tones and Break Indices) アノテーション付き
- 英語プロミネンス検出の**事実上の標準ベンチマーク**
- Kalinli & Narayanan (2007) 以降の全主要論文で使用

### 5.2 競合手法との比較

| システム | データセット | F1スコア | 出典 |
|---------|------------|---------|------|
| Kalinli (2007) | BURNC | 0.71 | IEEE TASLP |
| Rosenberg (2009) | BURNC | 0.73 | ICASSP |
| wav2vec2 + probe | BURNC | ~0.80 | 推定 |
| **本手法** | **BURNC** | **[要測定]** | - |

### 5.3 BURNC以外の選択肢

| データセット | 問題点 |
|------------|--------|
| CMU ARCTIC | プロミネンスラベルなし |
| LibriSpeech | 朗読音声のみ、アノテーションなし |
| 自作アノテーション | 査読で信頼性を疑われる |

### 5.4 結論

**BURNCなしでは査読付き論文での比較評価が不可能**

---

## 6. 投稿計画

### 6.1 ターゲット学会

1. **Interspeech 2026** (締切: 2026年3月頃)
   - 音声処理の最高峰
   - Short paper (4ページ)

2. **Human Augmentation Conference 2026**
   - デモ論文 (Visceral Resonance)
   - EMS連携の応用

### 6.2 主張する新規性

1. **リアルタイム因果処理**: 20ms以下のレイテンシ
2. **幾何平均Fusion**: ノイズロバストな特徴統合
3. **オンラインキャリブレーション**: 環境適応
4. **説明可能性**: 特徴ベースのフィードバック

### 6.3 予想される査読コメントへの対策

| 想定コメント | 対策 |
|-------------|------|
| 「DNNより精度低い」 | レイテンシ・説明可能性のトレードオフを主張 |
| 「重みは恣意的」 | Ablation study で各特徴の寄与を示す |
| 「ユーザー評価がない」 | 将来課題として認める |

---

## 7. 参考文献

- Oganian, Y., & Chang, E. F. (2019). A speech envelope landmark for syllable encoding. *Science Advances*, 5(11).
- Bello, J. P., et al. (2005). A tutorial on onset detection in music signals. *IEEE TSAP*, 13(5).
- Kalinli, O., & Narayanan, S. (2007). Prominence detection using auditory attention cues. *IEEE TASLP*, 17(5).
- Stevens, K. N. (1998). *Acoustic Phonetics*. MIT Press.
