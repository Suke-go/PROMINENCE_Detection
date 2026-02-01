/**
 * Prominence Detector - Wasm Implementation
 * 
 * WebAssembly版の音節プロミネンス検出器
 * Cライブラリ (syllable_detector.c) をWasmにコンパイルしたものを使用
 * 
 * 6つの特徴量を使用:
 *   1. Peak Rate - エンベロープ傾斜
 *   2. Spectral Flux - スペクトル変化
 *   3. High-Frequency Energy - 高周波エネルギー
 *   4. MFCC Delta - 音色変化
 *   5. Wavelet - 多スケール過渡検出
 *   6. Voiced Bonus - 有声性ボーナス
 */

class ProminenceDetectorWasm {
    constructor(options = {}) {
        this.config = {
            sampleRate: options.sampleRate || 48000,
            prominenceThreshold: options.prominenceThreshold || 0.2, // Low threshold for realtime mode
            minSyllableDistMs: options.minSyllableDistMs || 200,  // Slightly faster response
            calibrationDurationMs: options.calibrationDurationMs || 2000,
            minEnergyThreshold: options.minEnergyThreshold || 0.0001, // Lower energy gate
        };

        // Wasm module
        this.wasmModule = null;
        this.detector = null;
        this.isReady = false;
        this.isRunning = false;
        this.isCalibrating = false;

        // Audio
        this.audioContext = null;
        this.analyser = null;
        this.scriptProcessor = null;
        this.mediaStream = null;

        // State
        this.lastProminenceTime = 0;
        this.eventBuffer = new Float32Array(64 * 14); // 64 events, 14 floats each
        this.inputBuffer = null;

        // Callbacks
        this.onProminence = options.onProminence || (() => { });
        this.onFrame = options.onFrame || (() => { });
        this.onCalibrationStart = options.onCalibrationStart || (() => { });
        this.onCalibrationEnd = options.onCalibrationEnd || (() => { });
        this.onError = options.onError || ((err) => console.error(err));
        this.onReady = options.onReady || (() => { });
    }

    /**
     * Wasmモジュールを初期化
     */
    async init() {
        try {
            // Load Wasm module
            this.wasmModule = await SyllableModule();

            // Get function wrappers
            this._syllable_create = this.wasmModule.cwrap('syllable_create', 'number', ['number']);
            this._syllable_process = this.wasmModule.cwrap('syllable_process', 'number',
                ['number', 'number', 'number', 'number', 'number']);
            this._syllable_flush = this.wasmModule.cwrap('syllable_flush', 'number',
                ['number', 'number', 'number']);
            this._syllable_destroy = this.wasmModule.cwrap('syllable_destroy', null, ['number']);
            this._syllable_reset = this.wasmModule.cwrap('syllable_reset', null, ['number']);
            this._syllable_default_config = this.wasmModule.cwrap('syllable_default_config', 'number', ['number']);

            // Real-time mode API
            this._syllable_set_realtime_mode = this.wasmModule.cwrap('syllable_set_realtime_mode', null, ['number', 'number']);
            this._syllable_recalibrate = this.wasmModule.cwrap('syllable_recalibrate', null, ['number']);
            this._syllable_is_calibrating = this.wasmModule.cwrap('syllable_is_calibrating', 'number', ['number']);
            this._syllable_set_snr_threshold = this.wasmModule.cwrap('syllable_set_snr_threshold', null, ['number', 'number']);

            // Create detector with default config
            // Note: syllable_default_config returns a struct by value, which is complex in Wasm
            // For now, we pass NULL to use internal defaults
            this.detector = this._syllable_create(0);

            // Set lower SNR threshold BEFORE enabling realtime mode
            // Default is 6dB which is too aggressive, use -3dB for high sensitivity
            this._syllable_set_snr_threshold(this.detector, 6.0);

            // Enable real-time mode with geometric mean fusion
            this._syllable_set_realtime_mode(this.detector, 1);

            if (!this.detector) {
                throw new Error('Failed to create Wasm detector');
            }

            this.isReady = true;
            this.onReady();
            return true;
        } catch (err) {
            this.onError(err);
            return false;
        }
    }

    /**
     * マイク入力を開始
     */
    async start() {
        if (!this.isReady) {
            const success = await this.init();
            if (!success) return false;
        }

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: this.config.sampleRate
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Use ScriptProcessor for direct sample access (deprecated but necessary for real-time processing)
            const bufferSize = 1024; // ~21ms at 48kHz
            this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

            // Allocate input buffer in Wasm memory
            this.inputBuffer = this.wasmModule._malloc(bufferSize * 4); // float = 4 bytes

            this.scriptProcessor.onaudioprocess = (e) => this._processAudioBuffer(e);

            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);

            this.isRunning = true;

            // Start calibration
            this.startCalibration();

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

        if (this._calibrationCheckInterval) {
            clearInterval(this._calibrationCheckInterval);
            this._calibrationCheckInterval = null;
        }
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
        if (this.inputBuffer) {
            this.wasmModule._free(this.inputBuffer);
            this.inputBuffer = null;
        }
    }

    /**
     * キャリブレーション開始
     */
    startCalibration() {
        this.isCalibrating = true;
        this.onCalibrationStart();

        // Trigger C-side recalibration
        if (this._syllable_recalibrate) {
            this._syllable_recalibrate(this.detector);
        }

        // Poll C-side calibration state
        this._calibrationCheckInterval = setInterval(() => {
            const stillCalibrating = this._syllable_is_calibrating(this.detector);
            if (!stillCalibrating) {
                this._finishCalibration();
                clearInterval(this._calibrationCheckInterval);
            }
        }, 100);
    }

    /**
     * キャリブレーション完了
     */
    _finishCalibration() {
        this.isCalibrating = false;
        // Wasm detector handles calibration internally via adaptive thresholds
        this.onCalibrationEnd({ energy: 0, flux: 0, hf: 0 });
    }

    /**
     * 音声バッファ処理
     */
    _processAudioBuffer(e) {
        if (!this.isRunning || !this.detector || !this.wasmModule) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const numSamples = inputData.length;

        // Copy input to Wasm memory using setValue
        for (let i = 0; i < numSamples; i++) {
            this.wasmModule.setValue(this.inputBuffer + i * 4, inputData[i], 'float');
        }

        // Allocate event output buffer
        // SyllableEvent struct (68+ bytes with alignment):
        // - uint64_t timestamp_samples: 8 bytes (offset 0)
        // - double time_seconds: 8 bytes (offset 8)
        // - float peak_rate: 4 bytes (offset 16)
        // - float pr_slope: 4 bytes (offset 20)
        // - float energy: 4 bytes (offset 24)
        // - float f0: 4 bytes (offset 28)
        // - float delta_f0: 4 bytes (offset 32)
        // - float duration_s: 4 bytes (offset 36)
        // - float spectral_flux: 4 bytes (offset 40)
        // - float high_freq_energy: 4 bytes (offset 44)
        // - float mfcc_delta: 4 bytes (offset 48)
        // - float wavelet_score: 4 bytes (offset 52)
        // - float fusion_score: 4 bytes (offset 56)
        // - int onset_type: 4 bytes (offset 60)
        // - float prominence_score: 4 bytes (offset 64)
        // - int is_accented: 4 bytes (offset 68)
        // Total: 72 bytes
        const maxEvents = 8;
        const eventSize = 72; // bytes per event
        const eventBuffer = this.wasmModule._malloc(maxEvents * eventSize);

        // Process audio
        const numEvents = this._syllable_process(
            this.detector,
            this.inputBuffer,
            numSamples,
            eventBuffer,
            maxEvents
        );

        // DEBUG: Log processing stats periodically
        this._frameCount = (this._frameCount || 0) + 1;
        if (this._frameCount % 50 === 0) {  // Every ~1 second at 1024 samples/21ms
            // Calculate RMS of input audio
            let sumSq = 0;
            for (let j = 0; j < numSamples; j++) {
                sumSq += inputData[j] * inputData[j];
            }
            const rms = Math.sqrt(sumSq / numSamples);
            console.log(`[Wasm Debug] Frame ${this._frameCount}: numEvents=${numEvents}, calibrating=${this.isCalibrating}, RMS=${rms.toFixed(6)}`);
        }

        // Read events
        if (numEvents > 0) {
            for (let i = 0; i < numEvents; i++) {
                const basePtr = eventBuffer + i * eventSize;

                // Read fusion_score at offset 56
                const fusionScore = this.wasmModule.getValue(basePtr + 56, 'float');
                const energy = this.wasmModule.getValue(basePtr + 24, 'float');
                const spectralFlux = this.wasmModule.getValue(basePtr + 40, 'float');

                const event = {
                    timestamp: performance.now(),
                    fusionScore: fusionScore,
                    features: {
                        peakRate: this.wasmModule.getValue(basePtr + 16, 'float'),
                        energy: energy,
                        spectralFlux: spectralFlux,
                        highFreqEnergy: this.wasmModule.getValue(basePtr + 44, 'float'),
                        mfccDelta: this.wasmModule.getValue(basePtr + 48, 'float'),
                        f0: this.wasmModule.getValue(basePtr + 28, 'float')
                    },
                    prominenceScore: this.wasmModule.getValue(basePtr + 64, 'float') || 0,
                    isAccented: this.wasmModule.getValue(basePtr + 68, 'i32') !== 0
                };

                // DEBUG: Log every event
                console.log(`[Event ${i}] fusion=${fusionScore.toFixed(3)}, energy=${energy.toFixed(6)}, flux=${spectralFlux.toFixed(4)}, ` +
                    `threshold=${this.config.prominenceThreshold}, calibrating=${this.isCalibrating}`);

                if (this.isCalibrating) {
                    console.log(`  -> Skipped: still calibrating`);
                    continue;
                }

                const now = performance.now();
                const timeSinceLastProminence = now - this.lastProminenceTime;

                // Reject if energy is too low (noise) or fusion score below threshold
                const hasEnoughEnergy = event.features.energy > this.config.minEnergyThreshold ||
                    event.features.spectralFlux > 0.1;

                const passedThreshold = event.fusionScore > this.config.prominenceThreshold;
                const passedTiming = timeSinceLastProminence > this.config.minSyllableDistMs;

                console.log(`  -> passedThreshold=${passedThreshold}, passedTiming=${passedTiming}, hasEnoughEnergy=${hasEnoughEnergy}`);

                if (passedThreshold && passedTiming && hasEnoughEnergy) {
                    console.log(`  -> DETECTED!`);
                    this.lastProminenceTime = now;
                    this.onProminence(event);
                }
            }
        }

        // Free event buffer
        this.wasmModule._free(eventBuffer);

        // Call frame callback for visualization
        this.onFrame({
            raw: { energy: 0, flux: 0, hf: 0 },
            normalized: {
                energy: 0, flux: 0, hf: 0,
                scaledEnergy: 0, scaledFlux: 0, scaledHf: 0
            },
            fusionScore: 0,
            gates: {}
        });
    }

    /**
     * AudioContext を取得（音再生用）
     */
    getAudioContext() {
        return this.audioContext;
    }

    /**
     * Destroy detector
     */
    destroy() {
        this.stop();
        if (this.detector) {
            this._syllable_destroy(this.detector);
            this.detector = null;
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProminenceDetectorWasm;
}
