# Real-time Prominence Detection Demo

リアルタイムでマイク入力から音節のプロミネンス（強調）を検出するWebデモです。

![Demo Screenshot](docs/demo-screenshot.png)

## 🚀 クイックスタート

### 方法1: バッチファイルで起動（推奨）
```batch
start_server.bat
```
ダブルクリックするとブラウザが自動で開きます。

### 方法2: 手動でサーバー起動
```powershell
cd experiments/realtime_prominence/web_demo
python -m http.server 8000
```
ブラウザで http://localhost:8000 を開く

> ⚠️ **注意**: `file://` プロトコルではマイクアクセスがブロックされるため、必ずローカルサーバー経由でアクセスしてください。

---

## 📁 ファイル構成

```
web_demo/
├── index.html              # メインHTML
├── start_server.bat        # サーバー起動スクリプト
├── css/
│   └── style.css           # スタイルシート
├── js/
│   ├── prominence-detector-wasm.js  # 🔥 Wasm版検出器 (推奨)
│   ├── prominence-detector.js       # Pure JS版検出器
│   ├── demo.js                      # 🎨 デモUI
│   ├── syllable.js                  # Wasm glue code
│   └── syllable.wasm                # コンパイル済みWasm
└── README.md
```

### 検出器の選択

| ファイル | 説明 | 推奨度 |
|---------|------|--------|
| `prominence-detector-wasm.js` | C実装をWasmにコンパイル。高精度・低レイテンシ | ⭐ 推奨 |
| `prominence-detector.js` | Pure JavaScript実装。依存なし | 実験用 |

### スクリプトの分離

| ファイル | 役割 | 依存関係 |
|---------|------|----------|
| `prominence-detector-wasm.js` | Wasm版コアロジック | `syllable.js`, `syllable.wasm` |
| `prominence-detector.js` | Pure JS版コアロジック | なし（単独で使用可能） |
| `demo.js` | UI、視覚・聴覚フィードバック | どちらかのDetector |

---

## 🔧 コアアルゴリズム (`prominence-detector.js`)

### クラス: `ProminenceDetector`

```javascript
const detector = new ProminenceDetector({
    // 設定オプション
    fftSize: 512,
    noiseMultiplier: 3.0,      // ノイズフロアの何倍で検出
    prominenceThreshold: 0.60, // fusion score閾値
    minSyllableDistMs: 250,    // 最小検出間隔
    calibrationDurationMs: 2000,
    
    // コールバック
    onProminence: (event) => { /* 検出時 */ },
    onFrame: (data) => { /* 毎フレーム */ },
    onCalibrationStart: () => { /* キャリブ開始 */ },
    onCalibrationEnd: (noiseFloor) => { /* キャリブ完了 */ },
});

await detector.start();  // 開始
detector.stop();         // 停止
detector.startCalibration();  // 再キャリブレーション
```

### 特徴量

| 名前 | 説明 | 用途 |
|------|------|------|
| **Energy** | スペクトル全体のエネルギー | 音量レベル |
| **Spectral Flux** | 前フレームからのスペクトル変化量 | 音素境界検出 |
| **High Frequency** | 高周波帯域のエネルギー | 子音検出 |

### Fusion Score

```
fusionScore = 0.6 × max(E, F, HF) + 0.4 × weighted_avg(E, F, HF)
```

### 検出ゲート

| ゲート | 条件 | 目的 |
|--------|------|------|
| `fusion` | Score > 閾値 | 全体的な強度 |
| `energy` | E > ノイズ×3 | 十分なエネルギー |
| `flux` | F > ノイズ×3 | 明確な変化 |
| `fluxMax` | F < 0.92 | インパルス除外 |
| `impulse` | 履歴平均 > 0.05 | 瞬間ノイズ除外 |
| `interval` | 前回から > 250ms | 連続検出抑制 |

---

## 🎨 デモUI (`demo.js`)

### 機能

- **自動キャリブレーション**: 開始時に2秒間の無音を測定
- **視覚フィードバック**: 検出時に画面が黒→緑に変化
- **80Hz音フィードバック**: 検出強度に応じた音圧で再生
- **リアルタイムメトリクス**: E/F/HF値とメーター表示
- **イベントログ**: 検出履歴の表示

### UIコントロール

| 要素 | 説明 |
|------|------|
| `▶ マイクを開始` | 録音開始/停止 |
| `🔄 再キャリブ` | ノイズフロア再測定 |
| `🔊 80Hz音フィードバック` | 音のオン/オフ |

---

## 🔬 自分のプロジェクトで使う

### 最小限の使用例

```html
<script src="js/prominence-detector.js"></script>
<script>
const detector = new ProminenceDetector({
    onProminence: (event) => {
        console.log('検出!', event.fusionScore);
    }
});
detector.start();
</script>
```

### Node.js環境

```javascript
const ProminenceDetector = require('./js/prominence-detector.js');
// ※ ブラウザAPIに依存するため、完全なNode.js対応には追加実装が必要
```

---

## ⚙️ パラメータ調整

### 感度を上げる（より多く検出）
```javascript
{
    noiseMultiplier: 2.0,      // 2倍に下げる
    prominenceThreshold: 0.50, // 閾値を下げる
    minSyllableDistMs: 150,    // 間隔を短く
}
```

### 誤検出を減らす（より厳しく）
```javascript
{
    noiseMultiplier: 4.0,      // 4倍に上げる
    prominenceThreshold: 0.70, // 閾値を上げる
    minSyllableDistMs: 300,    // 間隔を長く
}
```

---

## 📊 技術仕様

- **サンプリング**: Web Audio API（ブラウザ依存、通常44.1kHz/48kHz）
- **FFTサイズ**: 512（約11ms @ 44.1kHz）
- **処理周期**: requestAnimationFrame（約16ms @ 60fps）
- **レイテンシ**: 理論値 約30-50ms

---

## 🔗 関連

- 親プロジェクト: [syllabledetection](../../README.md)
- Cライブラリ実装: [src/syllable_detector.c](../../src/syllable_detector.c)
