/**
 * Prominence Demo - UI & Visual Feedback
 * 
 * ProminenceDetector を使用したデモUI
 * 視覚・聴覚フィードバックを提供
 */

class ProminenceDemo {
    constructor() {
        this.detector = null;
        this.eventCount = 0;

        // DOM elements
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            calibrateBtn: document.getElementById('calibrateBtn'),
            soundToggle: document.getElementById('soundToggle'),
            indicator: document.getElementById('indicator'),
            indicatorText: document.getElementById('indicatorText'),
            status: document.getElementById('status'),
            controlPanel: document.getElementById('controlPanel'),
            noiseFloorDisplay: document.getElementById('noiseFloorDisplay'),
            metrics: document.getElementById('metrics'),
            eventLog: document.getElementById('eventLog'),
            energyValue: document.getElementById('energyValue'),
            fluxValue: document.getElementById('fluxValue'),
            hfValue: document.getElementById('hfValue'),
            energyMeter: document.getElementById('energyMeter'),
            fluxMeter: document.getElementById('fluxMeter'),
            hfMeter: document.getElementById('hfMeter'),
            countValue: document.getElementById('countValue'),
            nfEnergy: document.getElementById('nfEnergy'),
            nfFlux: document.getElementById('nfFlux'),
            nfHf: document.getElementById('nfHf'),
        };

        // Config
        this.config = {
            feedbackFrequency: 150,      // Hz
            feedbackDurationMs: 150,
            visualDurationMs: 400,
        };

        this._bindEvents();
    }

    _bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this._toggleMicrophone());
        this.elements.calibrateBtn.addEventListener('click', () => this._recalibrate());
    }

    _toggleMicrophone() {
        if (this.detector && this.detector.isRunning) {
            this._stop();
        } else {
            this._start();
        }
    }

    async _start() {
        // Check if Wasm mode is selected
        const wasmToggle = document.getElementById('wasmToggle');
        const useWasm = wasmToggle && wasmToggle.checked && typeof ProminenceDetectorWasm !== 'undefined';

        const DetectorClass = useWasm ? ProminenceDetectorWasm : ProminenceDetector;

        this.detector = new DetectorClass({
            onProminence: (event) => this._onProminence(event),
            onFrame: (data) => this._onFrame(data),
            onCalibrationStart: () => this._onCalibrationStart(),
            onCalibrationEnd: (noiseFloor) => this._onCalibrationEnd(noiseFloor),
            onError: (err) => this._onError(err),
            onReady: () => console.log(useWasm ? 'Wasm detector ready' : 'JS detector ready'),
        });

        const success = await this.detector.start();
        if (success) {
            this.elements.startBtn.textContent = '⏹ 停止';
            this.elements.startBtn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            this.elements.controlPanel.style.display = 'block';
            this.elements.metrics.style.display = 'grid';
            this.elements.eventLog.style.display = 'block';
        }
    }

    _stop() {
        if (this.detector) {
            this.detector.stop();
            this.detector = null;
        }

        this.elements.startBtn.textContent = '▶ マイクを開始';
        this.elements.startBtn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
        this.elements.status.textContent = 'マイクボタンをクリックして開始';
        this.elements.indicator.classList.remove('active', 'high');
        this.elements.indicator.style.background = '#1a1a1a';
        this.elements.indicatorText.textContent = '待機中';
        document.body.classList.remove('prominence-detected', 'prominence-high');
        this.elements.controlPanel.style.display = 'none';
        this.elements.noiseFloorDisplay.style.display = 'none';
    }

    _recalibrate() {
        if (this.detector && this.detector.isRunning && !this.detector.isCalibrating) {
            this.detector.startCalibration();
        }
    }

    _onCalibrationStart() {
        this.elements.status.textContent = '🔇 キャリブレーション中... 静かにしてください';
        this.elements.indicatorText.textContent = 'CALIB';
        this.elements.indicator.style.background = '#3b82f6';
    }

    _onCalibrationEnd(noiseFloor) {
        this.elements.nfEnergy.textContent = noiseFloor.energy.toFixed(4);
        this.elements.nfFlux.textContent = noiseFloor.flux.toFixed(4);
        this.elements.nfHf.textContent = noiseFloor.hf.toFixed(4);
        this.elements.noiseFloorDisplay.style.display = 'block';

        this.elements.status.textContent = '🎤 録音中... 話してください！';
        this.elements.indicatorText.textContent = '待機中';
        this.elements.indicator.style.background = '#1a1a1a';
    }

    _onFrame(data) {
        const { normalized } = data;

        this.elements.energyValue.textContent = normalized.scaledEnergy.toFixed(2);
        this.elements.fluxValue.textContent = normalized.scaledFlux.toFixed(2);
        this.elements.hfValue.textContent = normalized.scaledHf.toFixed(2);
        this.elements.energyMeter.style.width = (normalized.scaledEnergy * 100) + '%';
        this.elements.fluxMeter.style.width = (normalized.scaledFlux * 100) + '%';
        this.elements.hfMeter.style.width = (normalized.scaledHf * 100) + '%';
    }

    _onProminence(event) {
        this.eventCount++;
        this.elements.countValue.textContent = this.eventCount;

        // fusionScore の検証 (undefined/NaN 対策)
        const score = Number.isFinite(event.fusionScore) ? event.fusionScore : 0.5;

        // 音フィードバック
        this._playFeedbackTone(score);

        // 視覚フィードバック
        this._showVisualFeedback(score);

        // ログ
        this._logEvent(event);
    }

    _playFeedbackTone(fusionScore) {
        if (!this.elements.soundToggle.checked || !this.detector) return;

        const audioContext = this.detector.getAudioContext();
        if (!audioContext || audioContext.state !== 'running') return;

        // fusionScore の検証 (二重チェック)
        const safeScore = Number.isFinite(fusionScore) ? fusionScore : 0.5;

        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.value = this.config.feedbackFrequency;

        const gain = Math.min(0.5, Math.max(0.1, safeScore * 0.5));
        gainNode.gain.setValueAtTime(gain, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(
            0.01,
            audioContext.currentTime + this.config.feedbackDurationMs / 1000
        );

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.start();
        oscillator.stop(audioContext.currentTime + this.config.feedbackDurationMs / 1000);
    }

    _showVisualFeedback(fusionScore) {
        if (fusionScore > 0.8) {
            this.elements.indicator.classList.add('high');
            this.elements.indicator.classList.remove('active');
            document.body.classList.add('prominence-high');
            document.body.classList.remove('prominence-detected');
            this.elements.indicatorText.textContent = 'HIGH!';
        } else {
            this.elements.indicator.classList.add('active');
            this.elements.indicator.classList.remove('high');
            document.body.classList.add('prominence-detected');
            document.body.classList.remove('prominence-high');
            this.elements.indicatorText.textContent = 'DETECT';
        }

        setTimeout(() => {
            this.elements.indicator.classList.remove('active', 'high');
            document.body.classList.remove('prominence-detected', 'prominence-high');
            this.elements.indicatorText.textContent = '待機中';
        }, this.config.visualDurationMs);
    }

    _logEvent(event) {
        const timeStr = (event.timestamp / 1000).toFixed(2);
        // Handle both JS detector (scaledEnergy/scaledFlux/scaledHf) and Wasm detector formats
        const scaledEnergy = event.features?.scaledEnergy ?? event.features?.peakRate ?? 0;
        const scaledFlux = event.features?.scaledFlux ?? event.features?.spectralFlux ?? 0;
        const scaledHf = event.features?.scaledHf ?? event.features?.highFreqEnergy ?? 0;
        const fusionScore = event.fusionScore ?? 0;

        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        eventItem.innerHTML = `<span class="event-time">[${timeStr}s]</span> <span class="event-score">Score:${fusionScore.toFixed(2)}</span> E:${scaledEnergy.toFixed(2)} F:${scaledFlux.toFixed(2)} HF:${scaledHf.toFixed(2)}`;

        const firstItem = this.elements.eventLog.firstChild.nextSibling;
        this.elements.eventLog.insertBefore(eventItem, firstItem);

        // ログサイズ制限
        while (this.elements.eventLog.children.length > 20) {
            this.elements.eventLog.removeChild(this.elements.eventLog.lastChild);
        }
    }

    _onError(err) {
        this.elements.status.textContent = '❌ マイクへのアクセスが拒否されました';
        console.error(err);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.demo = new ProminenceDemo();
});
