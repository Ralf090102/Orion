/**
 * Auto-detects backend URL from the current window hostname.
 *
 * Priority:
 *  1. VITE_BACKEND_URL / VITE_BACKEND_WS env var (explicit override)
 *  2. window.location.hostname + :8000  (works on localhost AND any remote VM
 *     without changing .env — the same build just works everywhere)
 *  3. localhost:8000 fallback (SSR / Node context)
 */

function detectUrl(): { http: string; ws: string } {
	const envHttp = import.meta.env.VITE_BACKEND_URL;
	const envWs = import.meta.env.VITE_BACKEND_WS;

	if (envHttp && envWs) {
		return { http: envHttp, ws: envWs };
	}

	if (typeof window !== 'undefined') {
		const { protocol, hostname } = window.location;
		const wsProto = protocol === 'https:' ? 'wss' : 'ws';
		return {
			http: envHttp || `${protocol}//${hostname}:8000`,
			ws: envWs || `${wsProto}://${hostname}:8000`,
		};
	}

	// SSR fallback
	return {
		http: envHttp || 'http://localhost:8000',
		ws: envWs || 'ws://localhost:8000',
	};
}

const { http, ws } = detectUrl();

export const BACKEND_URL = http;
export const BACKEND_WS = ws;
