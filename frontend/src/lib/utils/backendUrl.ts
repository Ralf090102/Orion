/**
 * Auto-detects backend URL from the current window hostname.
 *
 * Priority:
 *  1. VITE_BACKEND_URL / VITE_BACKEND_WS env var (explicit override)
 *  2. Tauri desktop app detection -> localhost:8000 (backend managed by Tauri)
 *  3. window.location.hostname + :8000  (works on localhost AND any remote VM
 *     without changing .env — the same build just works everywhere)
 *  4. PUBLIC_BACKEND_URL / PUBLIC_BACKEND_WS (SSR fallback from SvelteKit)
 *  5. localhost:8000 fallback (final SSR fallback)
 */

// Get PUBLIC_ vars for SSR (these are set by SvelteKit from .env)
const PUBLIC_BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL as string | undefined;
const PUBLIC_BACKEND_WS = import.meta.env.PUBLIC_BACKEND_WS as string | undefined;

function detectUrl(): { http: string; ws: string } {
	const envHttp = import.meta.env.VITE_BACKEND_URL as string | undefined;
	const envWs = import.meta.env.VITE_BACKEND_WS as string | undefined;

	// Priority 1: Explicit VITE_ overrides
	if (envHttp && envWs) {
		return { http: envHttp, ws: envWs };
	}

	// Priority 2: Check if running in Tauri (desktop app)
	if (typeof window !== 'undefined' && '__TAURI__' in window) {
		// In Tauri, always use localhost:8000 as the backend is managed by Tauri
		return {
			http: envHttp || 'http://localhost:8000',
			ws: envWs || 'ws://localhost:8000',
		};
	}

	// Priority 3: Auto-detect from window.location (browser client-side)
	if (typeof window !== 'undefined') {
		const { protocol, hostname } = window.location;
		const wsProto = protocol === 'https:' ? 'wss' : 'ws';
		return {
			http: envHttp || `${protocol}//${hostname}:8000`,
			ws: envWs || `${wsProto}://${hostname}:8000`,
		};
	}

	// Priority 4 & 5: SSR fallback - use PUBLIC_ vars or localhost
	return {
		http: envHttp || PUBLIC_BACKEND_URL || 'http://localhost:8000',
		ws: envWs || PUBLIC_BACKEND_WS || 'ws://localhost:8000',
	};
}

const { http, ws } = detectUrl();

export const BACKEND_URL = http;
export const BACKEND_WS = ws;
