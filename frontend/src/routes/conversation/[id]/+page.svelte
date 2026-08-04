<script lang="ts">
	import ChatWindow from "$lib/components/chat/ChatWindow.svelte";
	import { pendingMessage } from "$lib/stores/pendingMessage";
	import { isAborted } from "$lib/stores/isAborted";
	import { onMount, onDestroy } from "svelte";
	import { page } from "$app/state";
	import { beforeNavigate, invalidateAll } from "$app/navigation";
	import { base } from "$app/paths";
	import { ERROR_MESSAGES, error } from "$lib/stores/errors";
	import { findCurrentModel } from "$lib/utils/models";
	import type { Message } from "$lib/types/Message";
	import file2base64 from "$lib/utils/file2base64";
	import { useSettingsStore } from "$lib/stores/settings.js";
	import { browser } from "$app/environment";
	import "katex/dist/katex.min.css";
	import { loading } from "$lib/stores/loading.js";
	import { WebSocketChat } from "$lib/utils/websocketChat";
	import titleUpdate from "$lib/stores/titleUpdate";
	
	// Voice mode imports
	import { 
		convoModeState, 
		initConversationMode,
		setStatus,
	} from "$lib/stores/conversationMode.svelte";
	import { 
		getVoiceModeController,
		stopVoiceModeController,
		type VoiceModeController,
	} from "$lib/utils/voiceMode";

	let { data = $bindable() } = $props();

	import { BACKEND_URL, BACKEND_WS } from '$lib/utils/backendUrl';
	const settings = useSettingsStore();
	
	let pending = $state(false);
	let files: File[] = $state([]);
	let messages = $state<Message[]>([]);
	let conversations = $state(data.conversations);
	let wsChat: WebSocketChat | null = null;
	let messageUpdateTrigger = $state(0); // Force reactivity trigger
	let hasPendingMessage = $state(false); // Track if we need to send pending message
	let pendingMessageContent = $state<string>('');
	let pendingMessageFiles = $state<File[]>([]);
	
	// Voice mode controller
	let voiceModeController: VoiceModeController | null = null;
	let voiceModeEnabled = $derived(convoModeState.enabled);

	$effect(() => {
		conversations = data.conversations;
	});
	
	// Watch voice mode state and start/stop controller
	$effect(() => {
		if (!browser) return;
		
		const sessionId = page.params.id;
		if (!sessionId) return;
		
		if (voiceModeEnabled && !voiceModeController) {
			// Start voice mode
			console.log('[VoiceMode] Starting voice mode controller');
			voiceModeController = getVoiceModeController({
				conversationId: sessionId,
				webSocketChat: wsChat,
				onTranscription: (text) => {
					console.log('[VoiceMode] Transcription received:', text);
				},
				onMessageSent: (userText) => {
					// Add user message to UI
					console.log('[VoiceMode] Adding user message to UI:', userText);
					const userMessage = createMessage('user', userText);
					messages = [...messages, userMessage];
					
					// Add empty assistant message that will be filled by WebSocket
					const assistantMessage = createMessage('assistant', '');
					messages = [...messages, assistantMessage];
					
					// Set loading state so ChatMessage can detect when it finishes
					$loading = true;
					pending = true;
				},
				onError: (errorMsg) => {
					console.error('[VoiceMode] Error:', errorMsg);
					$error = errorMsg;
				},
				onStatusChange: (status) => {
					console.log('[VoiceMode] Status changed:', status);
				},
			});
			
			voiceModeController.start().catch((err) => {
				console.error('[VoiceMode] Failed to start:', err);
				$error = 'Failed to start voice mode: ' + (err.message || 'Unknown error');
			});
		} else if (!voiceModeEnabled && voiceModeController) {
			// Stop voice mode
			console.log('[VoiceMode] Stopping voice mode controller');
			voiceModeController.stop();
			voiceModeController = null;
		}
	});
	
	// Keep voice mode controller updated with WebSocket reference
	$effect(() => {
		if (voiceModeController && wsChat) {
			voiceModeController.setWebSocket(wsChat);
		}
	});

	// UUID generation — crypto.randomUUID() only works in secure contexts (HTTPS/localhost)
	function generateId(): string {
		if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
			return crypto.randomUUID();
		}
		return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
			const r = (Math.random() * 16) | 0;
			return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
		});
	}

	// Simple message structure for local use
	function createMessage(from: 'user' | 'assistant', content: string, msgFiles?: any[]): Message {
		return {
			id: generateId(),
			from,
			content,
			files: msgFiles,
			createdAt: new Date(),
			updatedAt: new Date(),
			children: [],
			ancestors: [],
		} as Message;
	}

	// Fetch existing messages for this session
	async function loadMessages() {
		try {
			const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${page.params.id}`);
			if (!response.ok) {
				throw new Error('Failed to load session');
			}

			const sessionData = await response.json();
			
			// Convert backend messages to frontend format
			messages = sessionData.messages?.map((msg: any) => ({
				id: generateId(),
				from: msg.role === 'user' ? 'user' : 'assistant',
				content: msg.content,
				createdAt: new Date(msg.timestamp),
				updatedAt: new Date(msg.timestamp),
			})) || [];
		} catch (err) {
			console.error('Failed to load messages:', err);
			$error = 'Failed to load conversation';
		}
	}

	// Initialize WebSocket connection
	function initializeWebSocket() {
		const sessionId = page.params.id;
		if (!sessionId) return;
		
		if (wsChat) {
			wsChat.disconnect();
		}

		wsChat = new WebSocketChat({
			sessionId,
			backendUrl: BACKEND_WS,
			onMessage: (content, done) => {
				if (done) {
					$loading = false;
					pending = false;
					
					// Notify voice mode controller that response is complete
					if (voiceModeController) {
						voiceModeController.onResponseComplete();
					}
					return;
				}

				// Update the last assistant message with proper reactivity
				const lastIndex = messages.length - 1;
				const lastMsg = messages[lastIndex];
				
				if (lastMsg && lastMsg.from === 'assistant') {
					// Force reactivity increment
					messageUpdateTrigger++;
					
					// Create a completely new message object to trigger reactivity
					const updatedMessage: Message = {
						...lastMsg,
						content: lastMsg.content + content,
						updatedAt: new Date(),
						// Force Svelte to detect this as a different object
						_updateCount: messageUpdateTrigger
					} as Message;
					
					// Create new array with updated message
					messages = [
						...messages.slice(0, lastIndex),
						updatedMessage
					];
				} else {
					console.warn('[WebSocket] No assistant message to update');
				}
			},
			onSources: (sources) => {
				// Attach the RAG sources actually retrieved for this response to
				// the last assistant message, same reactivity pattern as onMessage.
				const lastIndex = messages.length - 1;
				const lastMsg = messages[lastIndex];

				if (lastMsg && lastMsg.from === 'assistant') {
					messageUpdateTrigger++;

					const updatedMessage: Message = {
						...lastMsg,
						sources,
						_updateCount: messageUpdateTrigger
					} as Message;

					messages = [
						...messages.slice(0, lastIndex),
						updatedMessage
					];
				}
			},
			onTitleGenerated: (title) => {
				console.log('[WebSocket] Title generated:', title);
				// Update sidebar via titleUpdate store
				const convId = page.params.id;
				if (convId) {
					$titleUpdate = {
						convId,
						title: title
					};
				}
			},
			onError: (errorMsg) => {
				console.error('[WebSocket] Error:', errorMsg);
				$error = errorMsg;
				$loading = false;
				pending = false;
			},
			onConnect: () => {
				console.log('[WebSocket] Connected to chat server');
				
				// Send pending message if exists and WebSocket is now ready
				if (hasPendingMessage && pendingMessageContent) {
					console.log('[WebSocket] Sending pending message after connection');
					// Use setTimeout to ensure the connection is fully established
					setTimeout(async () => {
						files = pendingMessageFiles;
						await writeMessage({ prompt: pendingMessageContent });
						hasPendingMessage = false;
						pendingMessageContent = '';
						pendingMessageFiles = [];
					}, 100);
				}
			},
			onDisconnect: () => {
				console.log('[WebSocket] Disconnected from chat server');
			}
		});

		wsChat.connect();
	}

	// Send a message through WebSocket
	async function writeMessage({ prompt }: { prompt?: string }): Promise<void> {
		if (!prompt || !wsChat) {
			console.warn('[WriteMessage] Missing prompt or WebSocket not initialized');
			return;
		}

		console.log('[WriteMessage] Sending message:', prompt);
		console.log('[WriteMessage] WebSocket connected:', wsChat?.isConnected());

		try {
			$isAborted = false;
			$loading = true;
			pending = true;

			// Convert files to base64 first
			const base64Files = await Promise.all(
				(files ?? []).map((file) =>
					file2base64(file).then((value) => ({
						type: "base64" as const,
						value,
						mime: file.type,
						name: file.name,
					}))
				)
			);

			// Add user message with proper file structure for display
			const userMessage = createMessage('user', prompt, base64Files.length > 0 ? base64Files : undefined);
			messages = [...messages, userMessage];

			// Add empty assistant message
			const assistantMessage = createMessage('assistant', '');
			messages = [...messages, assistantMessage];

			// Send via WebSocket
			wsChat.sendMessage(prompt, base64Files);
			files = [];

		} catch (err) {
			console.error('[WriteMessage] Error:', err);
			$error = (err as Error).message || ERROR_MESSAGES.default;
			$loading = false;
			pending = false;
		}
	}

	async function stopGeneration() {
		$isAborted = true;
		$loading = false;
		pending = false;
		// Optionally send stop message to backend
	}

	function handleKeydown(event: KeyboardEvent) {
		// Stop generation on ESC key when loading
		if (event.key === "Escape" && $loading) {
			event.preventDefault();
			stopGeneration();
		}
	}

	onMount(async () => {
		// Store pending message info to send after WebSocket connects
		if ($pendingMessage) {
			hasPendingMessage = true;
			pendingMessageContent = $pendingMessage.content;
			pendingMessageFiles = $pendingMessage.files || [];
			$pendingMessage = undefined;
			console.log('[Mount] Stored pending message, will send after WebSocket connects');
		}
	});

	// Reload messages and reconnect WebSocket when session ID changes
	$effect(() => {
		const sessionId = page.params.id;
		if (!browser || !sessionId) return;

		// Disconnect existing WebSocket before creating new one
		if (wsChat) {
			wsChat.disconnect();
			wsChat = null;
		}
		
		// Reset state
		messages = [];
		files = [];
		$loading = false;
		pending = false;

		// Load messages and initialize WebSocket for new/current session
		loadMessages();
		initializeWebSocket();
	});

	onDestroy(() => {
		// Cleanup WebSocket
		if (wsChat) {
			wsChat.disconnect();
		}
		
		// Cleanup voice mode controller
		if (voiceModeController) {
			voiceModeController.stop();
			voiceModeController = null;
		}
		stopVoiceModeController();
	});

	async function onMessage(content: string) {
		await writeMessage({ prompt: content });
	}

	async function onRetry(payload: { id: Message["id"]; content?: string }) {
		try {
			// Find the last user message
			const lastUserMessage = [...messages].reverse().find(msg => msg.from === 'user');
			if (!lastUserMessage) {
				console.warn('[Retry] No user message found');
				return;
			}
			
			const retryContent = payload.content || lastUserMessage.content;
			
			// Delete last 2 messages from database (user + assistant)
			const response = await fetch(
				`${BACKEND_URL}/api/chat/sessions/${page.params.id}/messages/last?count=2`,
				{ method: 'DELETE' }
			);
			
			if (!response.ok) {
				throw new Error('Failed to delete messages for retry');
			}
			
			// Remove last 2 messages from UI
			messages = messages.slice(0, -2);
			
			// Resend the message
			await writeMessage({ prompt: retryContent });
			
		} catch (err) {
			console.error('[Retry] Error:', err);
			$error = 'Failed to retry message';
		}
	}

	async function onShowAlternateMsg(payload: { id: Message["id"] }) {
		// Alternate messages not supported in simple version
		console.log('Alternate messages not supported yet');
	}

	beforeNavigate((navigation) => {
		if (!page.params.id) return;

		const navigatingAway =
			navigation.to?.route.id !== page.route.id || navigation.to?.params?.id !== page.params.id;

		if ($loading && navigatingAway) {
			// Stop generation when navigating away
			stopGeneration();
		}

		$isAborted = true;
		$loading = false;
	});

	let title = $derived.by(() => {
		const rawTitle = conversations.find((conv) => conv.id === page.params.id)?.title ?? '';
		return rawTitle ? rawTitle.charAt(0).toUpperCase() + rawTitle.slice(1) : 'Chat';
	});

	let currentModel = $derived(findCurrentModel(data.models, data.oldModels, data.model));
</script>

<svelte:window onkeydown={handleKeydown} />

<svelte:head>
	<title>{title}</title>
</svelte:head>

<ChatWindow
	loading={$loading}
	{pending}
	{messages}
	messagesAlternatives={[]}
	shared={false}
	preprompt={data.preprompt}
	bind:files
	onmessage={onMessage}
	onretry={onRetry}
	onshowAlternateMsg={onShowAlternateMsg}
	onstop={stopGeneration}
	models={data.models}
	{currentModel}
/>
