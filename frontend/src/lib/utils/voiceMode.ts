/**
 * Voice Mode Controller
 * 
 * Orchestrates the conversation mode flow:
 * 1. VAD detects speech and records audio
 * 2. Audio is sent to STT for transcription
 * 3. Transcribed text is sent via WebSocket with voice_mode flag
 * 4. (Auto-TTS handled separately by ChatMessage component)
 * 5. After TTS completes, resume listening (if autoResume enabled)
 */

import { browser } from "$app/environment";
import { api } from "$lib/api";
import { VoiceActivityDetector, type VADOptions } from "./vad";
import { 
	setStatus,
	getSettings,
	type ConvoModeStatus,
} from "$lib/stores/conversationMode.svelte";
import { interruptTTS } from "$lib/stores/ttsState.svelte";
import type { WebSocketChat } from "./websocketChat";

export interface VoiceModeControllerOptions {
	/** Conversation ID */
	conversationId: string;
	/** WebSocket chat instance for sending messages */
	webSocketChat: WebSocketChat | null;
	/** Callback when transcription completes (before sending to chat) */
	onTranscription?: (text: string) => void;
	/** Callback when an error occurs */
	onError?: (error: string) => void;
	/** Callback when status changes */
	onStatusChange?: (status: ConvoModeStatus) => void;
	/** Callback when volume changes (for visualization) */
	onVolumeChange?: (volume: number) => void;
}

export class VoiceModeController {
	private options: VoiceModeControllerOptions;
	private vad: VoiceActivityDetector | null = null;
	private isActive: boolean = false;
	private isProcessing: boolean = false;
	
	// For hold-to-talk mode
	private isHolding: boolean = false;
	
	constructor(options: VoiceModeControllerOptions) {
		this.options = options;
	}
	
	/**
	 * Start conversation mode (begin listening)
	 */
	async start(): Promise<void> {
		if (!browser) return;
		if (this.isActive) {
			console.warn('[VoiceMode] Already active');
			return;
		}
		
		const settings = getSettings();
		
		try {
			// Create VAD with settings
			const vadOptions: Partial<VADOptions> = {
				silenceThreshold: -50, // Could come from backend config
				silenceDuration: settings.silenceDuration,
				minSpeechDuration: 500,
				maxSpeechDuration: 60000,
				autoCalibrate: true,
				onSpeechStart: () => this.handleSpeechStart(),
				onSpeechEnd: (blob, duration) => this.handleSpeechEnd(blob, duration),
				onVolumeChange: (volume) => this.options.onVolumeChange?.(volume),
				onError: (error) => this.handleError(error),
				onCalibrationComplete: (noiseFloor) => {
					console.log(`[VoiceMode] Calibration complete, noise floor: ${noiseFloor.toFixed(1)}dB`);
				},
			};
			
			this.vad = new VoiceActivityDetector(vadOptions);
			
			// Start VAD
			await this.vad.start();
			this.isActive = true;
			
			// Update status based on input mode
			if (settings.inputMode === 'auto') {
				this.setStatus('listening');
			} else {
				this.setStatus('idle');
			}
			
			console.log(`[VoiceMode] Started in ${settings.inputMode} mode`);
		} catch (error) {
			const msg = error instanceof Error ? error.message : 'Failed to start voice mode';
			this.handleError(new Error(msg));
			throw error;
		}
	}
	
	/**
	 * Stop conversation mode
	 */
	stop(): void {
		if (!this.isActive) return;
		
		this.isActive = false;
		this.isHolding = false;
		
		if (this.vad) {
			this.vad.stop();
			this.vad = null;
		}
		
		this.setStatus('off');
		console.log('[VoiceMode] Stopped');
	}
	
	/**
	 * Pause listening (keep resources)
	 */
	pause(): void {
		if (this.vad) {
			this.vad.pause();
		}
		this.setStatus('idle');
	}
	
	/**
	 * Resume listening after pause
	 */
	resume(): void {
		if (!this.isActive || this.isProcessing) return;
		
		const settings = getSettings();
		
		if (this.vad) {
			this.vad.resume();
		}
		
		if (settings.inputMode === 'auto') {
			this.setStatus('listening');
		} else {
			this.setStatus('idle');
		}
	}
	
	/**
	 * Check if voice mode is active
	 */
	isRunning(): boolean {
		return this.isActive;
	}
	
	/**
	 * Check if currently processing (STT/LLM)
	 */
	isCurrentlyProcessing(): boolean {
		return this.isProcessing;
	}
	
	// ========== Push-to-Talk / Hold-to-Talk Controls ==========
	
	/**
	 * Start recording manually (for push-to-talk)
	 */
	startRecording(): void {
		if (!this.isActive || this.isProcessing) return;
		
		const settings = getSettings();
		if (settings.inputMode === 'auto') {
			console.warn('[VoiceMode] Cannot manually start recording in auto mode');
			return;
		}
		
		// For push-to-talk and hold-to-talk, we manually trigger recording
		// VAD will still detect when to stop (silence), or we stop manually
		if (this.vad) {
			this.vad.resume();
		}
		this.setStatus('listening');
		console.log('[VoiceMode] Manual recording started');
	}
	
	/**
	 * Stop recording manually (for push-to-talk)
	 */
	stopRecording(): void {
		if (!this.isActive) return;
		
		const settings = getSettings();
		if (settings.inputMode === 'auto') return;
		
		if (this.vad) {
			this.vad.forceStopRecording();
		}
		console.log('[VoiceMode] Manual recording stopped');
	}
	
	/**
	 * Hold started (for hold-to-talk)
	 */
	holdStart(): void {
		const settings = getSettings();
		if (settings.inputMode !== 'hold-to-talk') return;
		
		this.isHolding = true;
		this.startRecording();
	}
	
	/**
	 * Hold released (for hold-to-talk)
	 */
	holdEnd(): void {
		const settings = getSettings();
		if (settings.inputMode !== 'hold-to-talk' || !this.isHolding) return;
		
		this.isHolding = false;
		this.stopRecording();
	}
	
	// ========== Event handlers for external events ==========
	
	/**
	 * Called when TTS playback completes (from ChatMessage)
	 * Resume listening if autoResume is enabled
	 */
	onTTSComplete(): void {
		if (!this.isActive) return;
		
		const settings = getSettings();
		
		if (settings.autoResume && settings.inputMode === 'auto') {
			console.log('[VoiceMode] TTS complete, resuming listening');
			this.resume();
		} else {
			this.setStatus('idle');
		}
	}
	
	/**
	 * Called when user interrupts (starts speaking while TTS playing)
	 */
	onUserInterrupt(): void {
		// This would be called from the TTS player when it detects
		// the user speaking. For now, we handle it via VAD callbacks.
		console.log('[VoiceMode] User interrupt detected');
	}
	
	/**
	 * Update WebSocket reference (if it changes)
	 */
	setWebSocket(ws: WebSocketChat | null): void {
		this.options.webSocketChat = ws;
	}
	
	// ========== Private Methods ==========
	
	private handleSpeechStart(): void {
		console.log('[VoiceMode] Speech started');
		
		// Interrupt any playing TTS when user starts speaking
		interruptTTS();
		
		this.setStatus('listening');
	}
	
	private async handleSpeechEnd(audioBlob: Blob, durationMs: number): Promise<void> {
		console.log(`[VoiceMode] Speech ended, duration: ${durationMs}ms, size: ${audioBlob.size} bytes`);
		
		if (this.isProcessing) {
			console.warn('[VoiceMode] Already processing, ignoring');
			return;
		}
		
		this.isProcessing = true;
		this.setStatus('processing');
		
		try {
			// 1. Send to STT
			const text = await this.transcribeAudio(audioBlob);
			
			if (!text?.trim()) {
				console.log('[VoiceMode] Empty transcription, ignoring');
				this.resumeAfterProcessing();
				return;
			}
			
			console.log('[VoiceMode] Transcription:', text);
			this.options.onTranscription?.(text);
			
			// 2. Send to chat via WebSocket
			await this.sendToChat(text);
			
			// Status will change to 'speaking' when TTS starts
			// Then onTTSComplete will be called to resume listening
			
		} catch (error) {
			const msg = error instanceof Error ? error.message : 'Processing failed';
			console.error('[VoiceMode] Processing error:', msg);
			this.options.onError?.(msg);
			this.resumeAfterProcessing();
		}
	}
	
	private async transcribeAudio(audioBlob: Blob): Promise<string> {
		// Convert blob to File for the API
		const audioFile = new File([audioBlob], 'recording.webm', { 
			type: audioBlob.type || 'audio/webm' 
		});
		
		try {
			const result = await api.speech.transcribe(audioFile);
			return result.text;
		} catch (error) {
			console.error('[VoiceMode] STT error:', error);
			throw new Error('Speech recognition failed');
		}
	}
	
	private async sendToChat(text: string): Promise<void> {
		const ws = this.options.webSocketChat;
		
		if (!ws || !ws.isConnected()) {
			throw new Error('WebSocket not connected');
		}
		
		const settings = getSettings();
		
		// Send with voice mode flags
		ws.sendMessage(text, undefined, {
			voice_mode: true,
			disable_rag: settings.disableRAG,
			input_type: 'voice',
			// Use auto rag_mode unless explicitly disabled
			rag_mode: settings.disableRAG ? 'never' : 'auto',
			include_sources: !settings.disableRAG,
		});
		
		console.log('[VoiceMode] Message sent to chat', { 
			voice_mode: true, 
			disable_rag: settings.disableRAG 
		});
	}
	
	private resumeAfterProcessing(): void {
		this.isProcessing = false;
		
		if (!this.isActive) return;
		
		const settings = getSettings();
		
		// In auto mode, immediately resume listening
		// In other modes, wait for TTS to complete
		if (settings.inputMode === 'auto' && !settings.autoTTS) {
			// No TTS, resume immediately
			this.resume();
		} else {
			// TTS will play, onTTSComplete will resume
			this.setStatus('idle');
		}
	}
	
	private setStatus(status: ConvoModeStatus): void {
		setStatus(status);
		this.options.onStatusChange?.(status);
	}
	
	private handleError(error: Error): void {
		console.error('[VoiceMode] Error:', error.message);
		this.options.onError?.(error.message);
		
		// Reset to safe state
		this.isProcessing = false;
		if (this.isActive) {
			this.setStatus('idle');
		}
	}
}

// ========== Singleton Management ==========

let activeController: VoiceModeController | null = null;

/**
 * Get or create the voice mode controller for a conversation
 */
export function getVoiceModeController(options: VoiceModeControllerOptions): VoiceModeController {
	// If there's an active controller for a different conversation, stop it
	if (activeController && activeController !== null) {
		activeController.stop();
	}
	
	activeController = new VoiceModeController(options);
	return activeController;
}

/**
 * Get the current active controller (if any)
 */
export function getActiveVoiceModeController(): VoiceModeController | null {
	return activeController;
}

/**
 * Stop and cleanup the active controller
 */
export function stopVoiceModeController(): void {
	if (activeController) {
		activeController.stop();
		activeController = null;
	}
}

export default VoiceModeController;
