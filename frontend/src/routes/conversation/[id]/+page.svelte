<script lang="ts">
	import ChatWindow from "$lib/components/chat/ChatWindow.svelte";
	import { pendingMessage } from "$lib/stores/pendingMessage";
	import { isAborted } from "$lib/stores/isAborted";
	import { onMount, onDestroy } from "svelte";
	import { page } from "$app/state";
	import { beforeNavigate } from "$app/navigation";
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

	let { data = $bindable() } = $props();

	const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';
	const settings = useSettingsStore();
	
	let pending = $state(false);
	let files: File[] = $state([]);
	let messages = $state<Message[]>([]);
	let conversations = $state(data.conversations);
	let wsChat: WebSocketChat | null = null;

	$effect(() => {
		conversations = data.conversations;
	});

	// Simple message structure for local use
	function createMessage(from: 'user' | 'assistant', content: string, msgFiles?: any[]): Message {
		return {
			id: crypto.randomUUID(),
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
				id: crypto.randomUUID(),
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
		if (wsChat) {
			wsChat.disconnect();
		}

		console.log('[WebSocket] Initializing connection for session:', page.params.id);

		wsChat = new WebSocketChat({
			sessionId: page.params.id,
			onMessage: (content, done) => {
				console.log('[WebSocket] Received message:', { content, done });
				
				if (done) {
					console.log('[WebSocket] Generation complete');
					$loading = false;
					pending = false;
					return;
				}

				// Update the last assistant message with proper reactivity
				const lastIndex = messages.length - 1;
				const lastMsg = messages[lastIndex];
				
				if (lastMsg && lastMsg.from === 'assistant') {
					// Create a new message object to trigger reactivity
					const updatedMessage = {
						...lastMsg,
						content: lastMsg.content + content,
						updatedAt: new Date(),
					};
					
					// Create new array with updated message
					messages = [
						...messages.slice(0, lastIndex),
						updatedMessage
					];
					
					console.log('[WebSocket] Updated assistant message:', {
						length: updatedMessage.content.length,
						preview: updatedMessage.content.substring(0, 50),
						totalMessages: messages.length
					});
				} else {
					console.warn('[WebSocket] No assistant message to update, messages:', messages);
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

			// Add user message
			const userMessage = createMessage('user', prompt, files.length > 0 ? files : undefined);
			messages = [...messages, userMessage];
			console.log('[WriteMessage] Added user message, total messages:', messages.length);

			// Add empty assistant message
			const assistantMessage = createMessage('assistant', '');
			messages = [...messages, assistantMessage];
			console.log('[WriteMessage] Added empty assistant message, total messages:', messages.length);

			// Convert files to base64 if present
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

			if (base64Files.length > 0) {
				console.log('[WriteMessage] Converted files to base64:', base64Files.length);
			}

			// Send via WebSocket
			console.log('[WriteMessage] Sending to WebSocket...');
			wsChat.sendMessage(prompt, base64Files);
			files = [];
			console.log('[WriteMessage] Message sent successfully');

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
		// Load existing messages
		await loadMessages();

		// Initialize WebSocket
		initializeWebSocket();

		// Send pending message if exists
		if ($pendingMessage) {
			files = $pendingMessage.files;
			await writeMessage({ prompt: $pendingMessage.content });
			$pendingMessage = undefined;
		}
	});

	// Reload messages and reconnect WebSocket when session ID changes
	$effect(() => {
		const sessionId = page.params.id;
		if (!browser || !sessionId) return;

		console.log('[Effect] Session ID changed to:', sessionId);
		
		// Reset state
		messages = [];
		files = [];
		$loading = false;
		pending = false;

		// Reload for new session
		loadMessages();
		initializeWebSocket();
	});

	onDestroy(() => {
		if (wsChat) {
			wsChat.disconnect();
		}
	});

	async function onMessage(content: string) {
		await writeMessage({ prompt: content });
	}

	async function onRetry(payload: { id: Message["id"]; content?: string }) {
		// Simple retry: just resend the last user message
		const lastUserMessage = [...messages].reverse().find(msg => msg.from === 'user');
		if (lastUserMessage) {
			await writeMessage({ prompt: payload.content || lastUserMessage.content });
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
