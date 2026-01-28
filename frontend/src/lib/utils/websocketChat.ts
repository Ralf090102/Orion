/**
 * WebSocket chat client for Orion backend
 * Connects to FastAPI WebSocket endpoint at /ws/chat/{session_id}
 */

export interface WebSocketChatOptions {
	sessionId: string;
	onMessage: (content: string, done: boolean) => void;
	onError?: (error: string) => void;
	onConnect?: () => void;
	onDisconnect?: () => void;
	onTitleGenerated?: (title: string) => void;
	backendUrl?: string;
}

export class WebSocketChat {
	private ws: WebSocket | null = null;
	private options: WebSocketChatOptions;
	private reconnectAttempts = 0;
	private maxReconnectAttempts = 3;
	private reconnectDelay = 1000;
	private shouldReconnect = true;

	constructor(options: WebSocketChatOptions) {
		this.options = options;
	}

	connect() {
		const wsUrl = this.options.backendUrl || import.meta.env.PUBLIC_BACKEND_WS || 'ws://localhost:8000';
		const url = `${wsUrl}/ws/chat/${this.options.sessionId}`;

		// Reset reconnection flag when explicitly connecting
		this.shouldReconnect = true;

		try {
			this.ws = new WebSocket(url);

			this.ws.onopen = () => {
				console.log('WebSocket connected');
				this.reconnectAttempts = 0;
				this.options.onConnect?.();
			};

			this.ws.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					
					// Handle different message types from backend
					if (data.type === 'token') {
						// Streaming content token by token
						this.options.onMessage(data.content || '', false);
					} else if (data.type === 'done') {
						// Generation complete
						this.options.onMessage('', true);
					} else if (data.type === 'error') {
						// Error from backend
						this.options.onError?.(data.content || data.data?.message || 'Unknown error');
					} else if (data.type === 'connected') {
						// Connection acknowledged
						console.log('[WebSocket] Connected to server');
					} else if (data.type === 'sources') {
						// RAG sources received (could be used to display citations)
					} else if (data.type === 'metadata') {
						// Metadata received (processing time, RAG status, etc.)
					} else if (data.type === 'title') {
						// Session title generated - notify parent to update sidebar
						console.log('[WebSocket] Session title generated:', data.content);
						this.options.onTitleGenerated?.(data.content);
					} else if (data.type === 'pong') {
						// Pong response to ping
					} else {
						console.warn('[WebSocket] Unknown message type:', data.type);
					}
				} catch (err) {
					console.error('Failed to parse WebSocket message:', err, event.data);
					this.options.onError?.('Failed to parse server response');
				}
			};

			this.ws.onerror = (event) => {
				console.error('WebSocket error:', event);
				this.options.onError?.('WebSocket connection error');
			};

			this.ws.onclose = (event) => {
				console.log('WebSocket closed', event.code, event.reason);
				this.options.onDisconnect?.();

				// Only attempt to reconnect if shouldReconnect flag is true
				if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
					this.reconnectAttempts++;
					console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);
					setTimeout(() => this.connect(), this.reconnectDelay);
					this.reconnectDelay *= 2; // Exponential backoff
				}
			};
		} catch (err) {
			console.error('Failed to create WebSocket:', err);
			this.options.onError?.('Failed to create WebSocket connection');
		}
	}

	sendMessage(message: string, files?: File[]) {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			console.error('[WebSocket] Not connected, readyState:', this.ws?.readyState);
			this.options.onError?.('Not connected to server');
			return;
		}

		try {
			const payload = {
				type: 'message',
				content: message,
				data: {
					files: files || [],
					rag_mode: 'auto',
					include_sources: true
				}
			};
			
			console.log('[WebSocket] Sending:', payload);
			this.ws.send(JSON.stringify(payload));
		} catch (err) {
			console.error('[WebSocket] Failed to send message:', err);
			this.options.onError?.('Failed to send message');
		}
	}

	disconnect() {
		// Prevent any reconnection attempts
		this.shouldReconnect = false;
		this.reconnectAttempts = this.maxReconnectAttempts;
		
		if (this.ws) {
			this.ws.close();
			this.ws = null;
		}
	}

	isConnected(): boolean {
		return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
	}
}
