/**
 * Voice Activity Detection (VAD) using Web Audio API
 * 
 * Detects when a user starts and stops speaking, providing
 * recorded audio blobs for processing by STT.
 * 
 * Uses RMS (root mean square) of audio signal to detect activity.
 */

export interface VADOptions {
	/** dB level below which is considered silence (default: -50) */
	silenceThreshold: number;
	/** Milliseconds of silence before triggering speech end (default: 1500) */
	silenceDuration: number;
	/** Minimum speech duration in ms to be valid (default: 500) */
	minSpeechDuration: number;
	/** Maximum speech duration in ms before auto-stop (default: 60000) */
	maxSpeechDuration: number;
	/** Auto-calibrate noise floor on start (default: true) */
	autoCalibrate: boolean;
	/** Calibration duration in ms (default: 500) */
	calibrationDuration: number;
	/** Audio sample rate (default: 16000 for Whisper compatibility) */
	sampleRate: number;
	/** Called when speech is detected */
	onSpeechStart?: () => void;
	/** Called when speech ends with the recorded audio */
	onSpeechEnd?: (audioBlob: Blob, durationMs: number) => void;
	/** Called during recording with current volume level (0-1) */
	onVolumeChange?: (volume: number) => void;
	/** Called when an error occurs */
	onError?: (error: Error) => void;
	/** Called when calibration completes */
	onCalibrationComplete?: (noiseFloor: number) => void;
}

export type VADState = 'idle' | 'calibrating' | 'listening' | 'recording' | 'stopped';

const DEFAULT_OPTIONS: VADOptions = {
	silenceThreshold: -50,
	silenceDuration: 1500,
	minSpeechDuration: 500,
	maxSpeechDuration: 60000,
	autoCalibrate: true,
	calibrationDuration: 500,
	sampleRate: 16000,
};

/**
 * Convert linear amplitude (0-1) to decibels
 */
function amplitudeToDb(amplitude: number): number {
	if (amplitude <= 0) return -Infinity;
	return 20 * Math.log10(amplitude);
}

/**
 * Calculate RMS (root mean square) from audio data
 */
function calculateRMS(dataArray: Float32Array): number {
	let sum = 0;
	for (let i = 0; i < dataArray.length; i++) {
		sum += dataArray[i] * dataArray[i];
	}
	return Math.sqrt(sum / dataArray.length);
}

export class VoiceActivityDetector {
	private options: VADOptions;
	private state: VADState = 'idle';
	
	// Web Audio API
	private audioContext: AudioContext | null = null;
	private analyser: AnalyserNode | null = null;
	private mediaStream: MediaStream | null = null;
	private sourceNode: MediaStreamAudioSourceNode | null = null;
	
	// Recording
	private mediaRecorder: MediaRecorder | null = null;
	private recordedChunks: Blob[] = [];
	private recordingStartTime: number = 0;
	
	// VAD state
	private silenceStartTime: number = 0;
	private isSpeaking: boolean = false;
	private noiseFloor: number = -60; // Will be calibrated
	private effectiveThreshold: number;
	
	// Timers
	private analysisInterval: number | null = null;
	private maxDurationTimeout: number | null = null;
	private calibrationTimeout: number | null = null;
	private calibrationSamples: number[] = [];
	
	constructor(options: Partial<VADOptions> = {}) {
		this.options = { ...DEFAULT_OPTIONS, ...options };
		this.effectiveThreshold = this.options.silenceThreshold;
	}
	
	/**
	 * Get current VAD state
	 */
	getState(): VADState {
		return this.state;
	}
	
	/**
	 * Check if currently recording speech
	 */
	isRecording(): boolean {
		return this.state === 'recording';
	}
	
	/**
	 * Start VAD - requests microphone access and begins listening
	 */
	async start(): Promise<void> {
		if (this.state !== 'idle' && this.state !== 'stopped') {
			console.warn('[VAD] Already running, state:', this.state);
			return;
		}
		
		// Check for secure context (HTTPS or localhost) - required for mediaDevices
		if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
			const err = new Error('Microphone access requires a secure context (HTTPS). Voice mode is not available over HTTP.');
			console.error('[VAD] Secure context required:', err);
			this.options.onError?.(err);
			throw err;
		}
		
		try {
			// Request microphone access
			this.mediaStream = await navigator.mediaDevices.getUserMedia({
				audio: {
					sampleRate: this.options.sampleRate,
					channelCount: 1,
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true,
				},
			});
			
			// Create audio context
			this.audioContext = new AudioContext({
				sampleRate: this.options.sampleRate,
			});
			
			// Create analyser node for volume detection
			this.analyser = this.audioContext.createAnalyser();
			this.analyser.fftSize = 2048;
			this.analyser.smoothingTimeConstant = 0.3;
			
			// Connect media stream to analyser
			this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
			this.sourceNode.connect(this.analyser);
			
			// Setup MediaRecorder for capturing audio
			this.setupMediaRecorder();
			
			// Start calibration or listening
			if (this.options.autoCalibrate) {
				this.startCalibration();
			} else {
				this.startListening();
			}
		} catch (error) {
			const err = error instanceof Error ? error : new Error(String(error));
			console.error('[VAD] Failed to start:', err);
			this.options.onError?.(err);
			this.cleanup();
			throw err;
		}
	}
	
	/**
	 * Stop VAD completely
	 */
	stop(): void {
		this.state = 'stopped';
		
		// If currently recording, finalize it
		if (this.mediaRecorder?.state === 'recording') {
			this.mediaRecorder.stop();
		}
		
		this.cleanup();
	}
	
	/**
	 * Pause listening (keep resources, stop processing)
	 */
	pause(): void {
		if (this.analysisInterval) {
			window.clearInterval(this.analysisInterval);
			this.analysisInterval = null;
		}
		
		if (this.state === 'recording' && this.mediaRecorder?.state === 'recording') {
			this.mediaRecorder.pause();
		}
		
		this.state = 'idle';
	}
	
	/**
	 * Resume listening after pause
	 */
	resume(): void {
		if (this.state === 'stopped' || !this.audioContext) {
			console.warn('[VAD] Cannot resume - not initialized');
			return;
		}
		
		if (this.mediaRecorder?.state === 'paused') {
			this.mediaRecorder.resume();
			this.state = 'recording';
		} else {
			this.startListening();
		}
	}
	
	/**
	 * Force stop current recording and emit result
	 */
	forceStopRecording(): void {
		if (this.state === 'recording' && this.mediaRecorder?.state === 'recording') {
			this.mediaRecorder.stop();
		}
	}
	
	/**
	 * Update options dynamically
	 */
	updateOptions(newOptions: Partial<VADOptions>): void {
		this.options = { ...this.options, ...newOptions };
		this.effectiveThreshold = Math.max(
			this.options.silenceThreshold,
			this.noiseFloor + 6 // At least 6dB above noise floor
		);
	}
	
	// ========== Private Methods ==========
	
	private setupMediaRecorder(): void {
		if (!this.mediaStream) return;
		
		// Prefer webm/opus for better compression, fallback to wav
		const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
			? 'audio/webm;codecs=opus'
			: MediaRecorder.isTypeSupported('audio/webm')
				? 'audio/webm'
				: 'audio/wav';
		
		this.mediaRecorder = new MediaRecorder(this.mediaStream, {
			mimeType,
			audioBitsPerSecond: 128000,
		});
		
		this.mediaRecorder.ondataavailable = (event) => {
			if (event.data.size > 0) {
				this.recordedChunks.push(event.data);
			}
		};
		
		this.mediaRecorder.onstop = () => {
			this.handleRecordingComplete();
		};
		
		this.mediaRecorder.onerror = (event) => {
			console.error('[VAD] MediaRecorder error:', event);
			this.options.onError?.(new Error('MediaRecorder error'));
		};
	}
	
	private startCalibration(): void {
		this.state = 'calibrating';
		this.calibrationSamples = [];
		
		console.log('[VAD] Starting noise calibration...');
		
		// Collect samples during calibration period
		this.analysisInterval = window.setInterval(() => {
			const volume = this.getCurrentVolume();
			const db = amplitudeToDb(volume);
			if (isFinite(db)) {
				this.calibrationSamples.push(db);
			}
		}, 50);
		
		// End calibration after duration
		this.calibrationTimeout = window.setTimeout(() => {
			this.finishCalibration();
		}, this.options.calibrationDuration);
	}
	
	private finishCalibration(): void {
		if (this.analysisInterval) {
			window.clearInterval(this.analysisInterval);
			this.analysisInterval = null;
		}
		
		if (this.calibrationSamples.length > 0) {
			// Calculate average noise floor (use median to reduce outlier impact)
			const sorted = [...this.calibrationSamples].sort((a, b) => a - b);
			const median = sorted[Math.floor(sorted.length / 2)];
			this.noiseFloor = median;
			
			// Set effective threshold to be above noise floor
			this.effectiveThreshold = Math.max(
				this.options.silenceThreshold,
				this.noiseFloor + 6 // 6dB above noise floor
			);
			
			console.log(`[VAD] Calibration complete. Noise floor: ${this.noiseFloor.toFixed(1)}dB, Threshold: ${this.effectiveThreshold.toFixed(1)}dB`);
			this.options.onCalibrationComplete?.(this.noiseFloor);
		}
		
		this.startListening();
	}
	
	private startListening(): void {
		this.state = 'listening';
		this.isSpeaking = false;
		this.silenceStartTime = 0;
		
		// Start continuous analysis
		this.analysisInterval = window.setInterval(() => {
			this.analyzeAudio();
		}, 50); // 20Hz analysis rate
		
		console.log('[VAD] Listening for speech...');
	}
	
	private analyzeAudio(): void {
		const volume = this.getCurrentVolume();
		const db = amplitudeToDb(volume);
		
		// Report volume for visualizations
		this.options.onVolumeChange?.(volume);
		
		const isAboveThreshold = db > this.effectiveThreshold;
		const now = Date.now();
		
		if (isAboveThreshold) {
			// Sound detected
			if (!this.isSpeaking && this.state === 'listening') {
				// Speech started!
				this.startRecording();
			}
			// Reset silence timer
			this.silenceStartTime = 0;
		} else {
			// Silence detected
			if (this.isSpeaking) {
				if (this.silenceStartTime === 0) {
					// Start counting silence
					this.silenceStartTime = now;
				} else if (now - this.silenceStartTime >= this.options.silenceDuration) {
					// Silence duration exceeded - speech ended
					this.stopRecording();
				}
			}
		}
	}
	
	private getCurrentVolume(): number {
		if (!this.analyser) return 0;
		
		const dataArray = new Float32Array(this.analyser.fftSize);
		this.analyser.getFloatTimeDomainData(dataArray);
		return calculateRMS(dataArray);
	}
	
	private startRecording(): void {
		this.state = 'recording';
		this.isSpeaking = true;
		this.recordedChunks = [];
		this.recordingStartTime = Date.now();
		this.silenceStartTime = 0;
		
		// Start MediaRecorder
		if (this.mediaRecorder?.state === 'inactive') {
			this.mediaRecorder.start(100); // Collect data every 100ms
		}
		
		// Set max duration safety timeout
		this.maxDurationTimeout = window.setTimeout(() => {
			console.warn('[VAD] Max duration reached, stopping recording');
			this.forceStopRecording();
		}, this.options.maxSpeechDuration);
		
		console.log('[VAD] Speech started, recording...');
		this.options.onSpeechStart?.();
	}
	
	private stopRecording(): void {
		this.isSpeaking = false;
		
		// Clear max duration timeout
		if (this.maxDurationTimeout) {
			window.clearTimeout(this.maxDurationTimeout);
			this.maxDurationTimeout = null;
		}
		
		// Stop MediaRecorder (will trigger onstop -> handleRecordingComplete)
		if (this.mediaRecorder?.state === 'recording') {
			this.mediaRecorder.stop();
		}
	}
	
	private handleRecordingComplete(): void {
		const duration = Date.now() - this.recordingStartTime;
		
		console.log(`[VAD] Recording complete, duration: ${duration}ms, chunks: ${this.recordedChunks.length}`);
		
		// Check minimum duration
		if (duration < this.options.minSpeechDuration) {
			console.log('[VAD] Recording too short, discarding');
			this.recordedChunks = [];
			// Resume listening if not stopped
			if (this.state !== 'stopped') {
				this.startListening();
			}
			return;
		}
		
		// Create blob from recorded chunks
		if (this.recordedChunks.length > 0) {
			const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
			const audioBlob = new Blob(this.recordedChunks, { type: mimeType });
			
			console.log(`[VAD] Speech ended, blob size: ${audioBlob.size} bytes`);
			this.options.onSpeechEnd?.(audioBlob, duration);
		}
		
		this.recordedChunks = [];
		
		// Resume listening if not stopped
		if (this.state !== 'stopped') {
			this.startListening();
		}
	}
	
	private cleanup(): void {
		// Clear all timers
		if (this.analysisInterval) {
			window.clearInterval(this.analysisInterval);
			this.analysisInterval = null;
		}
		if (this.maxDurationTimeout) {
			window.clearTimeout(this.maxDurationTimeout);
			this.maxDurationTimeout = null;
		}
		if (this.calibrationTimeout) {
			window.clearTimeout(this.calibrationTimeout);
			this.calibrationTimeout = null;
		}
		
		// Stop media recorder
		if (this.mediaRecorder?.state !== 'inactive') {
			try {
				this.mediaRecorder?.stop();
			} catch {
				// Ignore errors when stopping
			}
		}
		this.mediaRecorder = null;
		
		// Disconnect audio nodes
		if (this.sourceNode) {
			this.sourceNode.disconnect();
			this.sourceNode = null;
		}
		
		// Close audio context
		if (this.audioContext?.state !== 'closed') {
			this.audioContext?.close();
		}
		this.audioContext = null;
		this.analyser = null;
		
		// Stop media stream tracks
		if (this.mediaStream) {
			this.mediaStream.getTracks().forEach(track => track.stop());
			this.mediaStream = null;
		}
		
		// Reset state
		this.recordedChunks = [];
		this.isSpeaking = false;
		this.silenceStartTime = 0;
		this.calibrationSamples = [];
		
		console.log('[VAD] Cleanup complete');
	}
}

// ========== Factory Function ==========

/**
 * Create a VAD instance with default options
 */
export function createVAD(options?: Partial<VADOptions>): VoiceActivityDetector {
	return new VoiceActivityDetector(options);
}

export default VoiceActivityDetector;
