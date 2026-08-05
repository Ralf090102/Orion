/**
 * Conversation Mode Store
 * 
 * Manages the state for voice conversation mode where users can speak
 * to the LLM and receive spoken responses via TTS.
 * 
 * State is persisted per-conversation in localStorage.
 */

import { browser } from "$app/environment";

// ========== Types ==========
export type ConvoModeStatus = 'off' | 'idle' | 'listening' | 'processing' | 'speaking';

export type InputMode = 'auto' | 'push-to-talk' | 'hold-to-talk';

export interface ConvoModeSettings {
	autoTTS: boolean;
	inputMode: InputMode;
	ttsVoice: string | null;  // null = use default
	silenceDuration: number;  // ms before auto-send
	autoResume: boolean;      // resume listening after TTS
	disableRAG: boolean;      // skip RAG pipeline for faster responses
	sttLanguage: string;      // Language code for STT (e.g., 'en', 'auto')
}

export interface ConvoModeState {
	enabled: boolean;
	status: ConvoModeStatus;
	settings: ConvoModeSettings;
}

// ========== Constants ==========
const STORAGE_KEY_PREFIX = 'orion-convo-mode-';

const DEFAULT_SETTINGS: ConvoModeSettings = {
	autoTTS: true,
	inputMode: 'auto',
	ttsVoice: null,
	silenceDuration: 1500,
	autoResume: true,
	disableRAG: false, // RAG on by default -- users can still disable it for faster voice responses
	sttLanguage: 'en', // Default to English for better accuracy
};

const DEFAULT_STATE: ConvoModeState = {
	enabled: false,
	status: 'off',
	settings: { ...DEFAULT_SETTINGS },
};

// ========== Reactive State ==========
// Current conversation ID being tracked
let currentConversationId = $state<string | null>(null);

// The actual conversation mode state - exported for direct reactive access
export const convoModeState = $state<ConvoModeState>({ ...DEFAULT_STATE });

// ========== LocalStorage Helpers ==========
function getStorageKey(conversationId: string): string {
	return `${STORAGE_KEY_PREFIX}${conversationId}`;
}

function loadFromStorage(conversationId: string): ConvoModeState | null {
	if (!browser) return null;
	
	try {
		const stored = localStorage.getItem(getStorageKey(conversationId));
		if (!stored) return null;
		
		const parsed = JSON.parse(stored);
		// Merge with defaults to handle missing fields from older versions
		return {
			...DEFAULT_STATE,
			...parsed,
			settings: {
				...DEFAULT_SETTINGS,
				...parsed.settings,
			},
		};
	} catch (e) {
		console.warn('[ConvoMode] Failed to load state from localStorage:', e);
		return null;
	}
}

function saveToStorage(conversationId: string, stateToSave: ConvoModeState): void {
	if (!browser) return;
	
	try {
		localStorage.setItem(
			getStorageKey(conversationId),
			JSON.stringify({
				enabled: stateToSave.enabled,
				settings: stateToSave.settings,
				// Don't persist status - always start fresh
			})
		);
	} catch (e) {
		console.warn('[ConvoMode] Failed to save state to localStorage:', e);
	}
}

// ========== Public API ==========

/**
 * Initialize/switch conversation mode for a specific conversation.
 * Loads persisted state from localStorage if available.
 */
export function initConversationMode(conversationId: string): void {
	currentConversationId = conversationId;
	
	const stored = loadFromStorage(conversationId);
	if (stored) {
		convoModeState.enabled = stored.enabled;
		convoModeState.status = stored.enabled ? 'idle' : 'off';
		convoModeState.settings = { ...DEFAULT_SETTINGS, ...stored.settings };
	} else {
		convoModeState.enabled = DEFAULT_STATE.enabled;
		convoModeState.status = DEFAULT_STATE.status;
		convoModeState.settings = { ...DEFAULT_SETTINGS };
	}
}

/**
 * Toggle conversation mode on/off.
 */
export function toggleConversationMode(): void {
	if (!currentConversationId) {
		console.warn('[ConvoMode] No conversation ID set');
		return;
	}
	
	convoModeState.enabled = !convoModeState.enabled;
	convoModeState.status = convoModeState.enabled ? 'idle' : 'off';
	
	saveToStorage(currentConversationId, convoModeState);
}

/**
 * Enable conversation mode.
 */
export function enableConversationMode(): void {
	if (!currentConversationId) return;
	
	convoModeState.enabled = true;
	convoModeState.status = 'idle';
	
	saveToStorage(currentConversationId, convoModeState);
}

/**
 * Disable conversation mode.
 */
export function disableConversationMode(): void {
	if (!currentConversationId) return;
	
	convoModeState.enabled = false;
	convoModeState.status = 'off';
	
	saveToStorage(currentConversationId, convoModeState);
}

/**
 * Update the current status (listening, processing, speaking, etc.)
 */
export function setStatus(newStatus: ConvoModeStatus): void {
	// Can't change status if disabled
	if (!convoModeState.enabled && newStatus !== 'off') {
		return;
	}
	convoModeState.status = newStatus;
}

/**
 * Update settings and persist to localStorage.
 */
export function updateSettings(newSettings: Partial<ConvoModeSettings>): void {
	if (!currentConversationId) return;
	
	convoModeState.settings = {
		...convoModeState.settings,
		...newSettings,
	};
	
	saveToStorage(currentConversationId, convoModeState);
}

/**
 * Reset settings to defaults.
 */
export function resetSettings(): void {
	if (!currentConversationId) return;
	
	convoModeState.settings = { ...DEFAULT_SETTINGS };
	saveToStorage(currentConversationId, convoModeState);
}

/**
 * Get current state (read-only access for components).
 */
export function getConvoModeState(): ConvoModeState {
	return convoModeState;
}

/**
 * Check if currently enabled.
 */
export function isEnabled(): boolean {
	return convoModeState.enabled;
}

/**
 * Get current status.
 */
export function getStatus(): ConvoModeStatus {
	return convoModeState.status;
}

/**
 * Get current settings.
 */
export function getSettings(): ConvoModeSettings {
	return convoModeState.settings;
}

// ========== Reactive Exports ==========
// Legacy alias - components should use convoModeState directly
// This getter-based export is kept for backward compatibility
export const conversationModeState = {
	get enabled() { return convoModeState.enabled; },
	get status() { return convoModeState.status; },
	get settings() { return convoModeState.settings; },
	get conversationId() { return currentConversationId; },
};
