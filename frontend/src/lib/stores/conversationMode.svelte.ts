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
	disableRAG: false,
};

const DEFAULT_STATE: ConvoModeState = {
	enabled: false,
	status: 'off',
	settings: { ...DEFAULT_SETTINGS },
};

// ========== Reactive State ==========
// Current conversation ID being tracked
let currentConversationId = $state<string | null>(null);

// The actual conversation mode state
let state = $state<ConvoModeState>({ ...DEFAULT_STATE });

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
		state = {
			...stored,
			status: stored.enabled ? 'idle' : 'off',
		};
	} else {
		state = { ...DEFAULT_STATE };
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
	
	state.enabled = !state.enabled;
	state.status = state.enabled ? 'idle' : 'off';
	
	saveToStorage(currentConversationId, state);
}

/**
 * Enable conversation mode.
 */
export function enableConversationMode(): void {
	if (!currentConversationId) return;
	
	state.enabled = true;
	state.status = 'idle';
	
	saveToStorage(currentConversationId, state);
}

/**
 * Disable conversation mode.
 */
export function disableConversationMode(): void {
	if (!currentConversationId) return;
	
	state.enabled = false;
	state.status = 'off';
	
	saveToStorage(currentConversationId, state);
}

/**
 * Update the current status (listening, processing, speaking, etc.)
 */
export function setStatus(newStatus: ConvoModeStatus): void {
	// Can't change status if disabled
	if (!state.enabled && newStatus !== 'off') {
		return;
	}
	state.status = newStatus;
}

/**
 * Update settings and persist to localStorage.
 */
export function updateSettings(newSettings: Partial<ConvoModeSettings>): void {
	if (!currentConversationId) return;
	
	state.settings = {
		...state.settings,
		...newSettings,
	};
	
	saveToStorage(currentConversationId, state);
}

/**
 * Reset settings to defaults.
 */
export function resetSettings(): void {
	if (!currentConversationId) return;
	
	state.settings = { ...DEFAULT_SETTINGS };
	saveToStorage(currentConversationId, state);
}

/**
 * Get current state (read-only access for components).
 */
export function getConvoModeState(): ConvoModeState {
	return state;
}

/**
 * Check if currently enabled.
 */
export function isEnabled(): boolean {
	return state.enabled;
}

/**
 * Get current status.
 */
export function getStatus(): ConvoModeStatus {
	return state.status;
}

/**
 * Get current settings.
 */
export function getSettings(): ConvoModeSettings {
	return state.settings;
}

// ========== Reactive Exports ==========
// Export reactive getters for use in components
export const conversationModeState = {
	get enabled() { return state.enabled; },
	get status() { return state.status; },
	get settings() { return state.settings; },
	get conversationId() { return currentConversationId; },
};
