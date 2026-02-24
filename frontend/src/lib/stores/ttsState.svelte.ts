/**
 * TTS Control Store
 * 
 * Simple global control for text-to-speech interruption.
 * Allows VoiceModeController to stop TTS when user starts speaking.
 * 
 * ChatMessage registers its stop callback, and voiceMode calls interruptTTS().
 */

// ========== Callback Registry ==========
// Currently active TTS stop callback (set by ChatMessage when TTS starts)
let activeStopCallback: (() => void) | null = null;

/**
 * Register the current TTS stop callback.
 * Called by ChatMessage when starting TTS.
 */
export function registerTTSStopCallback(callback: () => void): void {
	activeStopCallback = callback;
}

/**
 * Clear the TTS stop callback.
 * Called by ChatMessage when TTS finishes.
 */
export function clearTTSStopCallback(): void {
	activeStopCallback = null;
}

/**
 * Interrupt any currently playing TTS.
 * Called by VoiceModeController when user starts speaking.
 */
export function interruptTTS(): void {
	if (activeStopCallback) {
		console.log('[TTS] Interrupting playback');
		activeStopCallback();
		activeStopCallback = null;
	}
}

/**
 * Check if TTS is currently playing.
 */
export function isTTSPlaying(): boolean {
	return activeStopCallback !== null;
}
