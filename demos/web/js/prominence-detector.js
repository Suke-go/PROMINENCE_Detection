/**
 * Prominence Detector - Core Algorithm
 * 
 * リアルタイム音節プロミネンス検出のコアアルゴリズム
 * Web Audio API を使用してマイク入力から音響特徴量を抽出し、
 * プロミネンス（強調された音節）を検出します。
 * 
 * @author syllabledetection project
 * @license MIT
 */

class ProminenceDetector {
    constructor(options = {}) {
        // Configuration
        this.config = {
            fftSize: options.fftSize || 512,
            smoothingTimeConstant: options.smoothingTimeConstant || 0.5,
            minSyllableDistMs: options.minSyllableDistMs || 250,
            noiseMultiplier: options.noiseMultiplier || 3.0,
            fluxMaxThreshold: options.fluxMaxThreshold || 0.92,
            prominenceThreshold: options.prominenceThreshold || 0.60,
            highFreqCutoff: options.highFreqCutoff || 0.3,
            calibrationDurationMs: options.calibrationDurationMs || 2000,
            energyHistorySize: options.energyHistorySize || 4,
        };

        // State
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.isRunning = false;
        this.isCalibrating = false;
        this.lastProminenceTime = 0;
        this.prevSpectrum = null;

        // Noise floor (calibrated)
        this.noiseFloor = { energy: 0.01, flux: 0.01, hf: 0.01 };
        this.calibrationSamples = { energy: [], flux: [], hf: [] };

        // Energy history for impulse rejection
        this.energyHistory = [];

        // Callbacks
        this.onProminence = options.onProminence || (() => { });
        this.onFrame = options.onFrame || (() => { });
        this.onCalibrationStart = options.onCalibrationStart || (() => { });
        this.onCalibrationEnd = options.onCalibrationEnd || (() => { });
        this.onError = options.onError || ((err) => console.error(err));
    }

    /**
     * マイク入力を開始
     */
    async start() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = this.config.smoothingTimeConstant;

            source.connect(this.analyser);

            this.isRunning = true;
            this.prevSpectrum = null;
            this.energyHistory = [];

            // 自動キャリブレーション
            this.startCalibration();
            this._processAudio();

            return true;
        } catch (err) {
            this.onError(err);
            return false;
        }
    }

    /**
     * 停止
     */
    stop() {
        this.isRunning = false;
        this.isCalibrating = false;
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }

    /**
     * キャリブレーション開始
     */
    startCalibration() {
        this.isCalibrating = true;
        this.calibrationSamples = { energy: [], flux: [], hf: [] };
        this.onCalibrationStart();

        setTimeout(() => {
            this._finishCalibration();
        }, this.config.calibrationDurationMs);
    }

    /**
     * キャリブレーション完了
     */
    _finishCalibration() {
        this.isCalibrating = false;

        if (this.calibrationSamples.energy.length > 0) {
            const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
            this.noiseFloor.energy = Math.max(0.001, avg(this.calibrationSamples.energy));
            this.noiseFloor.flux = Math.max(0.001, avg(this.calibrationSamples.flux));
            this.noiseFloor.hf = Math.max(0.001, avg(this.calibrationSamples.hf));
        }

        this.onCalibrationEnd(this.noiseFloor);
    }

    /**
     * 音声処理ループ
     */
    _processAudio() {
        if (!this.isRunning) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const spectrum = new Float32Array(bufferLength);
        this.analyser.getFloatFrequencyData(spectrum);

        // dB -> linear 変換
        const linearSpectrum = spectrum.map(db => {
            const normalized = (db + 100) / 100;
            return Math.max(0, Math.min(1, normalized));
        });

        // 特徴量計算
        const features = this._computeFeatures(linearSpectrum, bufferLength);

        // キャリブレーション中はサンプル収集のみ
        if (this.isCalibrating) {
            this.calibrationSamples.energy.push(features.energy);
            this.calibrationSamples.flux.push(features.flux);
            this.calibrationSamples.hf.push(features.hf);
            requestAnimationFrame(() => this._processAudio());
            return;
        }

        // ノイズフロアからの相対値で正規化
        const normalized = this._normalizeFeatures(features);

        // プロミネンス検出
        const detection = this._detectProminence(normalized, features);

        // コールバック
        this.onFrame({
            raw: features,
            normalized: normalized,
            fusionScore: detection.fusionScore,
            gates: detection.gates
        });

        if (detection.detected) {
            this.lastProminenceTime = performance.now();
            this.onProminence({
                timestamp: this.lastProminenceTime,
                fusionScore: detection.fusionScore,
                features: normalized,
                gates: detection.gates
            });
        }

        requestAnimationFrame(() => this._processAudio());
    }

    /**
     * 特徴量計算
     */
    _computeFeatures(linearSpectrum, bufferLength) {
        // Energy
        const energy = linearSpectrum.reduce((sum, v) => sum + v * v, 0) / bufferLength;

        // Spectral Flux
        let flux = 0;
        if (this.prevSpectrum) {
            for (let i = 0; i < linearSpectrum.length; i++) {
                const diff = linearSpectrum[i] - this.prevSpectrum[i];
                if (diff > 0) flux += diff * diff;
            }
            flux = Math.sqrt(flux / linearSpectrum.length);
        }
        this.prevSpectrum = new Float32Array(linearSpectrum);

        // High Frequency Energy
        const cutoffBin = Math.floor(linearSpectrum.length * this.config.highFreqCutoff);
        let hf = 0;
        for (let i = cutoffBin; i < linearSpectrum.length; i++) {
            hf += linearSpectrum[i] * linearSpectrum[i];
        }
        hf = Math.sqrt(hf / (linearSpectrum.length - cutoffBin));

        return { energy, flux, hf };
    }

    /**
     * ノイズフロアベースの正規化
     */
    _normalizeFeatures(features) {
        const norm = (val, floor) => Math.max(0, (val - floor) / floor);
        const scale = (val) => Math.min(1, val / (this.config.noiseMultiplier * 2));

        const normEnergy = norm(features.energy, this.noiseFloor.energy);
        const normFlux = norm(features.flux, this.noiseFloor.flux);
        const normHf = norm(features.hf, this.noiseFloor.hf);

        return {
            energy: normEnergy,
            flux: normFlux,
            hf: normHf,
            scaledEnergy: scale(normEnergy),
            scaledFlux: scale(normFlux),
            scaledHf: scale(normHf)
        };
    }

    /**
     * プロミネンス検出
     */
    _detectProminence(normalized, raw) {
        const { scaledEnergy, scaledFlux, scaledHf } = normalized;
        const { energy: normEnergy, flux: normFlux } = normalized;

        // Fusion score
        const maxFeature = Math.max(scaledEnergy, scaledFlux, scaledHf);
        const avgFeature = scaledEnergy * 0.4 + scaledFlux * 0.35 + scaledHf * 0.25;
        const fusionScore = 0.6 * maxFeature + 0.4 * avgFeature;

        // Impulse rejection
        this.energyHistory.push(scaledEnergy);
        if (this.energyHistory.length > this.config.energyHistorySize) {
            this.energyHistory.shift();
        }
        const avgHistEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;

        const now = performance.now();
        const timeSinceLastProminence = now - this.lastProminenceTime;

        // Gate checks
        const gates = {
            fusion: fusionScore > this.config.prominenceThreshold,
            energy: normEnergy > this.config.noiseMultiplier,
            flux: normFlux > this.config.noiseMultiplier,
            fluxMax: scaledFlux < this.config.fluxMaxThreshold,
            impulse: avgHistEnergy > 0.05,
            interval: timeSinceLastProminence > this.config.minSyllableDistMs
        };

        const detected = Object.values(gates).every(v => v);

        return { detected, fusionScore, gates };
    }

    /**
     * AudioContext を取得（音再生用）
     */
    getAudioContext() {
        return this.audioContext;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProminenceDetector;
}
