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

		wsChat = new WebSocketChat({
			sessionId: page.params.id,
			onMessage: (content, done) => {
				if (done) {
					$loading = false;
					pending = false;
					return;
				}

				// Update the last assistant message
				const lastMsg = messages[messages.length - 1];
				if (lastMsg && lastMsg.from === 'assistant') {
					lastMsg.content += content;
					messages = [...messages]; // Trigger reactivity
				}
			},
			onError: (errorMsg) => {
				$error = errorMsg;
				$loading = false;
				pending = false;
			},
			onConnect: () => {
				console.log('Connected to chat');
			},
			onDisconnect: () => {
				console.log('Disconnected from chat');
			}
		});

		wsChat.connect();
	}

	// Send a message through WebSocket
	async function writeMessage({ prompt }: { prompt?: string }): Promise<void> {
		if (!prompt || !wsChat) return;

		try {
			$isAborted = false;
			$loading = true;
			pending = true;

			// Add user message
			const userMessage = createMessage('user', prompt, files.length > 0 ? files : undefined);
			messages = [...messages, userMessage];

			// Add empty assistant message
			const assistantMessage = createMessage('assistant', '');
			messages = [...messages, assistantMessage];

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

			// Send via WebSocket
			wsChat.sendMessage(prompt, base64Files);
			files = [];

		} catch (err) {
			$error = (err as Error).message || ERROR_MESSAGES.default;
			console.error(err);
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
				page.params.id,
				{
					base,
					inputs: prompt,
					messageId,
					isRetry,
					files: isRetry ? userMessage?.files : base64Files,
					selectedMcpServerNames: $enabledServers.map((s) => s.name),
					selectedMcpServers: $enabledServers.map((s) => ({
						name: s.name,
						url: s.url,
						headers: s.headers,
					})),
				},
				messageUpdatesAbortController.signal
			).catch((err) => {
				error.set(err.message);
			});
			if (messageUpdatesIterator === undefined) return;

			files = [];
			let buffer = "";
			// Initialize lastUpdateTime outside the loop to persist between updates
			let lastUpdateTime = new Date();

			for await (const update of messageUpdatesIterator) {
				if ($isAborted) {
					messageUpdatesAbortController.abort();
					return;
				}

				// Remove null characters added due to remote keylogging prevention
				// See server code for more details
				if (update.type === MessageUpdateType.Stream) {
					update.token = update.token.replaceAll("\0", "");
				}

				const isKeepAlive =
					update.type === MessageUpdateType.Status &&
					update.status === MessageUpdateStatus.KeepAlive;

				if (!isKeepAlive) {
					if (update.type === MessageUpdateType.Stream) {
						const existingUpdates = messageToWriteTo.updates ?? [];
						const lastUpdate = existingUpdates.at(-1);
						if (lastUpdate?.type === MessageUpdateType.Stream) {
							// Create fresh objects/arrays so the UI reacts to merged tokens
							const merged = {
								...lastUpdate,
								token: (lastUpdate.token ?? "") + (update.token ?? ""),
							};
							messageToWriteTo.updates = [...existingUpdates.slice(0, -1), merged];
						} else {
							messageToWriteTo.updates = [...existingUpdates, update];
						}
					} else {
						messageToWriteTo.updates = [...(messageToWriteTo.updates ?? []), update];
					}
				}
				const currentTime = new Date();

				// If we receive a non-stream update (e.g. tool/status/final answer),
				// flush any buffered stream tokens so the UI doesn't appear to cut
				// mid-sentence while tools are running or the final answer arrives.
				if (
					update.type !== MessageUpdateType.Stream &&
					!$settings.disableStream &&
					buffer.length > 0
				) {
					messageToWriteTo.content += buffer;
					buffer = "";
					lastUpdateTime = currentTime;
				}

				if (update.type === MessageUpdateType.Stream && !$settings.disableStream) {
					buffer += update.token;
					// Check if this is the first update or if enough time has passed
					if (currentTime.getTime() - lastUpdateTime.getTime() > updateDebouncer.maxUpdateTime) {
						messageToWriteTo.content += buffer;
						buffer = "";
						lastUpdateTime = currentTime;
					}
					pending = false;
				} else if (update.type === MessageUpdateType.FinalAnswer) {
					// Mirror server-side merge behavior so the UI reflects the
					// final text once tools complete, while preserving any
					// pre‑tool streamed content when appropriate.
					const hadTools =
						messageToWriteTo.updates?.some((u) => u.type === MessageUpdateType.Tool) ?? false;

					if (hadTools) {
						const existing = messageToWriteTo.content;
						const finalText = update.text ?? "";
						const trimmedExistingSuffix = existing.replace(/\s+$/, "");
						const trimmedFinalPrefix = finalText.replace(/^\s+/, "");
						const alreadyStreamed =
							finalText &&
							(existing.endsWith(finalText) ||
								(trimmedFinalPrefix.length > 0 &&
									trimmedExistingSuffix.endsWith(trimmedFinalPrefix)));

						if (existing && existing.length > 0) {
							if (alreadyStreamed) {
								// A. Already streamed the same final text; keep as-is.
								messageToWriteTo.content = existing;
							} else if (
								finalText &&
								(finalText.startsWith(existing) ||
									(trimmedExistingSuffix.length > 0 &&
										trimmedFinalPrefix.startsWith(trimmedExistingSuffix)))
							) {
								// B. Final text already includes streamed prefix; use it verbatim.
								messageToWriteTo.content = finalText;
							} else {
								// C. Merge with a paragraph break for readability.
								const needsGap = !/\n\n$/.test(existing) && !/^\n/.test(finalText ?? "");
								messageToWriteTo.content = existing + (needsGap ? "\n\n" : "") + finalText;
							}
						} else {
							messageToWriteTo.content = finalText;
						}
					} else {
						// No tools: final answer replaces streamed content so
						// the provider's final text is authoritative.
						messageToWriteTo.content = update.text ?? "";
					}
				} else if (
					update.type === MessageUpdateType.Status &&
					update.status === MessageUpdateStatus.Error
				) {
					// Check if this is a 402 payment required error
					if (update.statusCode === 402) {
						showSubscribeModal = true;
					} else {
						$error = update.message ?? "An error has occurred";
					}
				} else if (update.type === MessageUpdateType.Title) {
					const convInData = conversations.find(({ id }) => id === page.params.id);
					if (convInData) {
						convInData.title = update.title;

						$titleUpdate = {
							title: update.title,
							convId: page.params.id,
						};
					}
				} else if (update.type === MessageUpdateType.File) {
					messageToWriteTo.files = [
						...(messageToWriteTo.files ?? []),
						{ type: "hash", value: update.sha, mime: update.mime, name: update.name },
					];
				} else if (update.type === MessageUpdateType.RouterMetadata) {
					// Update router metadata immediately when received
					messageToWriteTo.routerMetadata = {
						route: update.route,
						model: update.model,
					};
				}
			}
		} catch (err) {
			if (err instanceof Error && err.message.includes("overloaded")) {
				$error = "Too much traffic, please try again.";
			} else if (err instanceof Error && err.message.includes("429")) {
				$error = ERROR_MESSAGES.rateLimited;
			} else if (err instanceof Error) {
				$error = err.message;
			} else {
				$error = ERROR_MESSAGES.default;
			}
			console.error(err);
		} finally {
			$loading = false;
			pending = false;
			await invalidateAll();
		}
	}

	async function stopGeneration() {
		await fetch(`${base}/conversation/${page.params.id}/stop-generating`, {
			method: "POST",
		}).then((r) => {
			if (r.ok) {
				setTimeout(() => {
					$isAborted = true;
					$loading = false;
				}, 500);
			} else {
				$isAborted = true;
				$loading = false;
			}
		});
	}

	function handleKeydown(event: KeyboardEvent) {
		// Stop generation on ESC key when loading
		if (event.key === "Escape" && $loading) {
			event.preventDefault();
			stopGeneration();
		}
	}

	onMount(async () => {
		if ($pendingMessage) {
			files = $pendingMessage.files;
			await writeMessage({ prompt: $pendingMessage.content });
			$pendingMessage = undefined;
		}

		const streaming = isConversationStreaming(messages);
		if (streaming) {
			addBackgroundGeneration({ id: page.params.id, startedAt: Date.now() });
			$loading = true;
		}
	});

	async function onMessage(content: string) {
		await writeMessage({ prompt: content });
	}

	async function onRetry(payload: { id: Message["id"]; content?: string }) {
		if (requireAuthUser()) return;

		const lastMsgId = payload.id;
		messagesPath = createMessagesPath(messages, lastMsgId);

		await writeMessage({
			prompt: payload.content,
			messageId: payload.id,
			isRetry: true,
		});
	}

	async function onShowAlternateMsg(payload: { id: Message["id"] }) {
		const msgId = payload.id;
		messagesPath = createMessagesPath(messages, msgId);
	}

	const settings = useSettingsStore();
	let messages = $state(data.messages);
	$effect(() => {
		messages = data.messages;
	});

	function isConversationStreaming(msgs: Message[]): boolean {
		const lastAssistant = [...msgs].reverse().find((msg) => msg.from === "assistant");
		if (!lastAssistant) return false;
		const hasFinalAnswer =
			lastAssistant.updates?.some((update) => update.type === MessageUpdateType.FinalAnswer) ??
			false;
		const hasError =
			lastAssistant.updates?.some(
				(update) =>
					update.type === MessageUpdateType.Status && update.status === MessageUpdateStatus.Error
			) ?? false;
		return !hasFinalAnswer && !hasError;
	}

	$effect(() => {
		const streaming = isConversationStreaming(messages);
		if (streaming) {
			$loading = true;
		} else if (!pending) {
			$loading = false;
		}

		if (!streaming && browser) {
			removeBackgroundGeneration(page.params.id);
		}
	});

	// create a linear list of `messagesPath` from `messages` that is a tree of threaded messages
	let messagesPath = $derived(createMessagesPath(messages));
	let messagesAlternatives = $derived(createMessagesAlternatives(messages));

	$effect(() => {
		if (browser && messagesPath.at(-1)?.id) {
			localStorage.setItem("leafId", messagesPath.at(-1)?.id as string);
		}
	});

	beforeNavigate((navigation) => {
		if (!page.params.id) return;

		const navigatingAway =
			navigation.to?.route.id !== page.route.id || navigation.to?.params?.id !== page.params.id;

		if ($loading && navigatingAway) {
			addBackgroundGeneration({ id: page.params.id, startedAt: Date.now() });
		}

		$isAborted = true;
		$loading = false;
	});

	let title = $derived.by(() => {
		const rawTitle = conversations.find((conv) => conv.id === page.params.id)?.title ?? data.title;
		return rawTitle ? rawTitle.charAt(0).toUpperCase() + rawTitle.slice(1) : rawTitle;
	});
</script>

<svelte:window onkeydown={handleKeydown} />

<svelte:head>
	<title>{title}</title>
</svelte:head>

<ChatWindow
	loading={$loading}
	{pending}
	messages={messagesPath as Message[]}
	{messagesAlternatives}
	shared={data.shared}
	preprompt={data.preprompt}
	bind:files
	onmessage={onMessage}
	onretry={onRetry}
	onshowAlternateMsg={onShowAlternateMsg}
	onstop={stopGeneration}
	models={data.models}
	currentModel={findCurrentModel(data.models, data.oldModels, data.model)}
/>

{#if showSubscribeModal}
	<SubscribeModal close={() => (showSubscribeModal = false)} />
{/if}
