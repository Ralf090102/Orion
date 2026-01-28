/**
 * Typed API client utilities for Orion backend
 * Simple fetch wrappers with proper error handling
 */

const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

/**
 * API error class for better error handling
 */
export class APIError extends Error {
	constructor(
		message: string,
		public status: number,
		public data?: unknown
	) {
		super(message);
		this.name = 'APIError';
	}
}

/**
 * Type definitions for API responses
 */
export interface Session {
	session_id: string;
	created_at: string;
	updated_at: string;
	message_count: number;
	metadata: Record<string, unknown>;
	messages?: Array<{
		role: 'user' | 'assistant';
		content: string;
		timestamp: string;
	}>;
}

export interface SessionResponse {
	status: string;
	message: string;
	session: Session;
	session_id?: string;
}

export interface SessionListResponse {
	status: string;
	sessions: Session[];
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
	endpoint: string,
	options: RequestInit = {}
): Promise<T> {
	const url = `${BACKEND_URL}${endpoint}`;
	
	try {
		const response = await fetch(url, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...options.headers,
			},
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => null);
			throw new APIError(
				errorData?.detail || `HTTP ${response.status}: ${response.statusText}`,
				response.status,
				errorData
			);
		}

		return await response.json();
	} catch (err) {
		if (err instanceof APIError) {
			throw err;
		}
		throw new APIError(
			err instanceof Error ? err.message : 'Network error',
			0
		);
	}
}

/**
 * API client methods
 */
export const api = {
	// Session management
	sessions: {
		create: (metadata?: Record<string, unknown>) =>
			fetchAPI<SessionResponse>('/api/chat/sessions', {
				method: 'POST',
				body: JSON.stringify({ metadata }),
			}),

		get: (sessionId: string) =>
			fetchAPI<SessionResponse>(`/api/chat/sessions/${sessionId}`),

		list: () =>
			fetchAPI<SessionListResponse>('/api/chat/sessions'),

		delete: (sessionId: string) =>
			fetchAPI<{ status: string }>(`/api/chat/sessions/${sessionId}`, {
				method: 'DELETE',
			}),

		update: (sessionId: string, data: { metadata?: Record<string, unknown> }) =>
			fetchAPI<{ session: unknown }>(`/api/chat/sessions/${sessionId}`, {
				method: 'PATCH',
				body: JSON.stringify(data),
			}),
	},

	// Message management
	messages: {
		delete: (sessionId: string, messageId: string) =>
			fetchAPI<{ status: string }>(`/api/chat/sessions/${sessionId}/messages/${messageId}`, {
				method: 'DELETE',
			}),
	},

	// RAG queries
	rag: {
		query: (query: string, options?: { k?: number; enable_reranking?: boolean }) =>
			fetchAPI<{ results: unknown[] }>('/api/query', {
				method: 'POST',
				body: JSON.stringify({ query, ...options }),
			}),

		ask: (query: string, options?: { k?: number; include_sources?: boolean }) =>
			fetchAPI<{ answer: string; sources?: unknown[] }>('/api/ask', {
				method: 'POST',
				body: JSON.stringify({ query, ...options }),
			}),
	},

	// Ingestion
	ingestion: {
		ingest: (path: string, clearExisting = false) =>
			fetchAPI<{ task_id: string }>('/api/ingest', {
				method: 'POST',
				body: JSON.stringify({ path, clear_existing: clearExisting }),
			}),

		status: (taskId: string) =>
			fetchAPI<{ status: string; progress: number }>(`/api/ingest/tasks/${taskId}`),

		clear: () =>
			fetchAPI<{ status: string }>('/api/ingest/clear', {
				method: 'POST',
			}),
	},

	// Health & Config
	health: () => fetchAPI<{ status: string }>('/health'),
	
	config: () => fetchAPI<Record<string, unknown>>('/api/config'),
};

/**
 * Get WebSocket URL for chat
 */
export function getChatWebSocketURL(sessionId: string): string {
	const wsUrl = import.meta.env.PUBLIC_BACKEND_WS || 'ws://localhost:8000';
	return `${wsUrl}/ws/chat/${sessionId}`;
}
