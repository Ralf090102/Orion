<script lang="ts">
	import type { Message } from "$lib/types/Message";
	import { tick } from "svelte";

	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";
	const publicConfig = usePublicConfig();
	import CopyToClipBoardBtn from "../CopyToClipBoardBtn.svelte";
	import IconLoading from "../icons/IconLoading.svelte";
	import CarbonRotate360 from "~icons/carbon/rotate-360";
	import CarbonTextToSpeech from "~icons/carbon/ibm-watson-text-to-speech";
	// import CarbonDownload from "~icons/carbon/download";

	import CarbonPen from "~icons/carbon/pen";
	import UploadedFile from "./UploadedFile.svelte";

	import MarkdownRenderer from "./MarkdownRenderer.svelte";
	import OpenReasoningResults from "./OpenReasoningResults.svelte";
	import Alternatives from "./Alternatives.svelte";
	import MessageAvatar from "./MessageAvatar.svelte";
	import { PROVIDERS_HUB_ORGS } from "@huggingface/inference";
	import { requireAuthUser } from "$lib/utils/auth";
	import ToolUpdate from "./ToolUpdate.svelte";
	import { isMessageToolUpdate } from "$lib/utils/messageUpdates";
	import { MessageUpdateType, type MessageToolUpdate } from "$lib/types/MessageUpdate";
	
	// Import conversation mode for auto-TTS
	import { convoModeState, setStatus } from "$lib/stores/conversationMode.svelte";
	import { getActiveVoiceModeController } from "$lib/utils/voiceMode";
	import { registerTTSStopCallback, clearTTSStopCallback } from "$lib/stores/ttsState.svelte";

	// Global TTS state shared across all message instances
	let globalTTSState = $state<{
		audio: HTMLAudioElement | null;
		messageId: string | null;
		audioUrl: string | null;
	}>({ audio: null, messageId: null, audioUrl: null });

	interface Props {
		message: Message;
		loading?: boolean;
		isAuthor?: boolean;
		readOnly?: boolean;
		isTapped?: boolean;
		alternatives?: Message["id"][];
		editMsdgId?: Message["id"] | null;
		isLast?: boolean;
		onretry?: (payload: { id: Message["id"]; content?: string }) => void;
		onshowAlternateMsg?: (payload: { id: Message["id"] }) => void;
	}

	let {
		message,
		loading = false,
		isAuthor: _isAuthor = true,
		readOnly: _readOnly = false,
		isTapped = $bindable(false),
		alternatives = [],
		editMsdgId = $bindable(null),
		isLast = false,
		onretry,
		onshowAlternateMsg,
	}: Props = $props();

	let contentEl: HTMLElement | undefined = $state();
	let isCopied = $state(false);
	let messageWidth: number = $state(0);
	let messageInfoWidth: number = $state(0);
	let isTTSLoading = $state(false);
	// AbortController for the active streaming fetch
	let ttsStreamController: AbortController | null = null;
	
	// Computed property to check if this message is currently playing
	let isReadingAloud = $derived(globalTTSState.messageId === message.id);
	
	// Track previous loading state to detect completion
	let prevLoading = $state(false);
	let autoTTSTriggered = $state(false);
	let initialLoadingSet = $state(false);

	import { BACKEND_URL } from '$lib/utils/backendUrl';

	// Auto-TTS: When conversation mode is enabled and this assistant message finishes loading, auto-play TTS
	$effect(() => {
		// Initialize prevLoading on first run
		if (!initialLoadingSet) {
			prevLoading = loading;
			initialLoadingSet = true;
			console.log('[ChatMessage] Initial loading set to:', loading, 'for message:', message.id);
			return;
		}
		
		// Detect when loading transitions from true to false (message just completed)
		const justFinishedLoading = prevLoading && !loading;
		
		// Log every state change
		if (prevLoading !== loading) {
			console.log('[ChatMessage] Loading changed:', prevLoading, '->', loading, 'for message:', message.id, 'isLast:', isLast);
		}
		
		prevLoading = loading;
		
		// Only auto-trigger once per message completion
		if (!justFinishedLoading || autoTTSTriggered) return;
		
		// Debug logging for auto-TTS conditions
		console.log('[ChatMessage] Auto-TTS check:', {
			messageId: message.id,
			from: message.from,
			isLast,
			enabled: convoModeState.enabled,
			autoTTS: convoModeState.settings?.autoTTS,
		});
		
		// Check conditions for auto-TTS:
		// 1. Must be an assistant message
		// 2. Must be the last message
		// 3. Conversation mode must be enabled
		// 4. autoTTS setting must be on
		if (
			message.from === 'assistant' &&
			isLast &&
			convoModeState.enabled &&
			convoModeState.settings.autoTTS
		) {
			console.log('[ChatMessage] Auto-TTS triggered for message:', message.id);
			autoTTSTriggered = true;
			
			// Update status to speaking
			setStatus('speaking');
			
			// Trigger TTS (small delay to ensure UI updates)
			setTimeout(() => {
				readAloud();
			}, 100);
		}
	});
	
	// Reset auto-TTS triggered flag when message ID changes
	$effect(() => {
		void message.id;
		autoTTSTriggered = false;
	});

	$effect(() => {
		// referenced to appease linter for currently-unused props
		void _isAuthor;
		void _readOnly;
	});
	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
			editFormEl?.requestSubmit();
		}
		if (e.key === "Escape") {
			editMsdgId = null;
		}
	}

	function handleCopy(event: ClipboardEvent) {
		if (!contentEl) return;

		const selection = window.getSelection();
		if (!selection || selection.isCollapsed) return;
		if (!selection.anchorNode || !selection.focusNode) return;

		const anchorInside = contentEl.contains(selection.anchorNode);
		const focusInside = contentEl.contains(selection.focusNode);
		if (!anchorInside && !focusInside) return;

		if (!event.clipboardData) return;

		const range = selection.getRangeAt(0);
		const wrapper = document.createElement("div");
		wrapper.appendChild(range.cloneContents());

		wrapper.querySelectorAll("[data-exclude-from-copy]").forEach((el) => {
			el.remove();
		});

		wrapper.querySelectorAll("*").forEach((el) => {
			el.removeAttribute("style");
			el.removeAttribute("class");
			el.removeAttribute("color");
			el.removeAttribute("bgcolor");
			el.removeAttribute("background");

			for (const attr of Array.from(el.attributes)) {
				if (attr.name === "id" || attr.name.startsWith("data-")) {
					el.removeAttribute(attr.name);
				}
			}
		});

		const html = wrapper.innerHTML;
		const text = wrapper.textContent ?? "";

		event.preventDefault();
		event.clipboardData.setData("text/html", html);
		event.clipboardData.setData("text/plain", text);
	}

	async function readAloud() {
		console.log('[ChatMessage] readAloud called for message:', message.id, {
			isReadingAloud,
			isTTSLoading,
			hasContent: !!contentWithoutThink?.trim(),
		});
		
		// ── Stop if already playing ──────────────────────────────────────
		if (isReadingAloud || isTTSLoading) {
			console.log('[ChatMessage] Stopping existing TTS playback');
			ttsStreamController?.abort();
			ttsStreamController = null;
			globalTTSState.audio?.pause();
			if (globalTTSState.audioUrl) URL.revokeObjectURL(globalTTSState.audioUrl);
			globalTTSState = { audio: null, messageId: null, audioUrl: null };
			isTTSLoading = false;
			return;
		}

		const textToRead = contentWithoutThink;
		if (!textToRead?.trim()) {
			console.log('[ChatMessage] No text to read');
			return;
		}

		console.log('[ChatMessage] Starting TTS for text length:', textToRead.length);
		isTTSLoading = true;

		// ── Set up abort controller for this session ─────────────────────
		const ac = new AbortController();
		ttsStreamController = ac;
		let stopped = false;

		function stopAll(revokeQueued: string[]) {
			stopped = true;
			ac.abort();
			globalTTSState.audio?.pause();
			if (globalTTSState.audioUrl) URL.revokeObjectURL(globalTTSState.audioUrl);
			for (const u of revokeQueued) URL.revokeObjectURL(u);
			globalTTSState = { audio: null, messageId: null, audioUrl: null };
			isTTSLoading = false;
			clearTTSStopCallback();  // Unregister from global TTS interrupt
		}

		// ── Fetch the streaming endpoint ─────────────────────────────────
		let response: Response;
		try {
			response = await fetch(`${BACKEND_URL}/api/speech/synthesize-stream`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text: textToRead, voice: null, format: 'wav' }),
				signal: ac.signal,
			});
			if (!response.ok || !response.body) {
				const err = await response.json().catch(() => ({ detail: 'TTS failed' }));
				throw new Error(err.detail || 'TTS request failed');
			}
		} catch (err: any) {
			if (err?.name !== 'AbortError') console.error('Read aloud failed:', err);
			isTTSLoading = false;
			ttsStreamController = null;
			return;
		}

		// ── Audio queue player ───────────────────────────────────────────
		// Each entry is an object-URL for a WAV blob
		const audioQueue: string[] = [];
		let isPlayingQueue = false;
		
		// Register stop callback so VoiceModeController can interrupt TTS
		registerTTSStopCallback(() => stopAll(audioQueue));

		function playNext() {
			if (stopped || audioQueue.length === 0) {
				isPlayingQueue = false;
				if (!stopped) {
					globalTTSState = { audio: null, messageId: null, audioUrl: null };
					// TTS completed naturally - notify voice mode controller
					notifyTTSComplete();
				}
				return;
			}
			isPlayingQueue = true;
			const url = audioQueue.shift()!;
			const audio = new Audio(url);
			globalTTSState = { audio, messageId: message.id, audioUrl: url };
			audio.onended = () => { URL.revokeObjectURL(url); playNext(); };
			audio.onerror = () => { URL.revokeObjectURL(url); playNext(); };
			audio.play().catch(() => playNext());
		}
		
		// Helper to notify voice mode controller when TTS completes
		function notifyTTSComplete() {
			console.log('[ChatMessage] TTS completed, notifying voice mode');
			
			// Clear the stop callback registration
			clearTTSStopCallback();
			
			// Update conversation mode status
			if (convoModeState.enabled) {
				setStatus('idle');
			}
			
			// Notify voice mode controller to resume listening
			const controller = getActiveVoiceModeController();
			console.log('[ChatMessage] Voice mode controller:', controller ? 'found' : 'not found');
			if (controller) {
				controller.onTTSComplete();
			}
		}

		// ── Read NDJSON stream ───────────────────────────────────────────
		const reader = response.body!.getReader();
		const decoder = new TextDecoder();
		let buf = '';
		isTTSLoading = false;  // first byte received — clear spinner

		try {
			while (!stopped) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				const lines = buf.split('\n');
				buf = lines.pop()!;  // last partial line stays in buffer

				for (const line of lines) {
					if (!line.trim() || stopped) continue;
					const msg = JSON.parse(line) as {
						audio_b64?: string; done?: boolean; error?: string; index?: number;
					};
					if (msg.error) throw new Error(`Synthesis failed: ${msg.error}`);
					if (msg.done) { stopped = true; break; }
					if (msg.audio_b64) {
						const bytes = Uint8Array.from(atob(msg.audio_b64), c => c.charCodeAt(0));
						const blob = new Blob([bytes], { type: 'audio/wav' });
						audioQueue.push(URL.createObjectURL(blob));
						if (!isPlayingQueue) playNext();
					}
				}
			}
		} catch (err: any) {
			if (err?.name !== 'AbortError') console.error('Read aloud stream error:', err);
			stopAll(audioQueue);
			return;
		} finally {
			ttsStreamController = null;
		}

		// Stream done — let any remaining queued chunks finish playing
		if (!isPlayingQueue && audioQueue.length > 0) playNext();
		if (!isPlayingQueue && audioQueue.length === 0) {
			globalTTSState = { audio: null, messageId: null, audioUrl: null };
			// TTS completed naturally (no audio chunks were queued)
			notifyTTSComplete();
		}
	}

	let editContentEl: HTMLTextAreaElement | undefined = $state();
	let editFormEl: HTMLFormElement | undefined = $state();

	// Zero-config reasoning autodetection: detect <think> blocks in content
	const THINK_BLOCK_REGEX = /(<think>[\s\S]*?(?:<\/think>|$))/gi;
	// Non-global version for .test() calls to avoid lastIndex side effects
	const THINK_BLOCK_TEST_REGEX = /(<think>[\s\S]*?(?:<\/think>|$))/i;
	let hasClientThink = $derived(message.content.split(THINK_BLOCK_REGEX).length > 1);

	// Strip think blocks for clipboard copy (always, regardless of detection)
	let contentWithoutThink = $derived.by(() =>
		message.content.replace(THINK_BLOCK_REGEX, "").trim()
	);

	type Block =
		| { type: "text"; content: string }
		| { type: "tool"; uuid: string; updates: MessageToolUpdate[] };

	type ToolBlock = Extract<Block, { type: "tool" }>;

	let blocks = $derived.by(() => {
		const updates = message.updates ?? [];
		const res: Block[] = [];
		const hasTools = updates.some(isMessageToolUpdate);
		let contentCursor = 0;
		let sawFinalAnswer = false;

		// Fast path: no tool updates at all
		if (!hasTools && updates.length === 0) {
			if (message.content) return [{ type: "text" as const, content: message.content }];
			return [];
		}

		for (const update of updates) {
			if (update.type === MessageUpdateType.Stream) {
				const token =
					typeof update.token === "string" && update.token.length > 0 ? update.token : null;
				const len = token !== null ? token.length : (update.len ?? 0);
				const chunk =
					token ??
					(message.content ? message.content.slice(contentCursor, contentCursor + len) : "");
				contentCursor += len;
				if (!chunk) continue;
				const last = res.at(-1);
				if (last?.type === "text") last.content += chunk;
				else res.push({ type: "text" as const, content: chunk });
			} else if (isMessageToolUpdate(update)) {
				const existingBlock = res.find(
					(b): b is ToolBlock => b.type === "tool" && b.uuid === update.uuid
				);
				if (existingBlock) {
					existingBlock.updates.push(update);
				} else {
					res.push({ type: "tool" as const, uuid: update.uuid, updates: [update] });
				}
			} else if (update.type === MessageUpdateType.FinalAnswer) {
				sawFinalAnswer = true;
				const finalText = update.text ?? "";
				const currentText = res
					.filter((b) => b.type === "text")
					.map((b) => (b as { type: "text"; content: string }).content)
					.join("");

				let addedText = "";
				if (finalText.startsWith(currentText)) {
					addedText = finalText.slice(currentText.length);
				} else if (!currentText.endsWith(finalText)) {
					const needsGap = !/\n\n$/.test(currentText) && !/^\n/.test(finalText);
					addedText = (needsGap ? "\n\n" : "") + finalText;
				}

				if (addedText) {
					const last = res.at(-1);
					if (last?.type === "text") {
						last.content += addedText;
					} else {
						res.push({ type: "text" as const, content: addedText });
					}
				}
			}
		}

		// If content remains unmatched (e.g., persisted stream markers), append the remainder
		// Skip when a FinalAnswer already provided the authoritative text.
		if (!sawFinalAnswer && message.content && contentCursor < message.content.length) {
			const remaining = message.content.slice(contentCursor);
			if (remaining.length > 0) {
				const last = res.at(-1);
				if (last?.type === "text") last.content += remaining;
				else res.push({ type: "text" as const, content: remaining });
			}
		} else if (!res.some((b) => b.type === "text") && message.content) {
			// Fallback: no text produced at all
			res.push({ type: "text" as const, content: message.content });
		}

		return res;
	});

	$effect(() => {
		if (isCopied) {
			setTimeout(() => {
				isCopied = false;
			}, 1000);
		}
	});

	let editMode = $derived(editMsdgId === message.id);
	$effect(() => {
		if (editMode) {
			tick();
			if (editContentEl) {
				editContentEl.value = message.content;
				editContentEl?.focus();
			}
		}
	});
</script>

{#if message.from === "assistant"}
	<div
		bind:offsetWidth={messageWidth}
		class="group relative -mb-4 flex w-fit max-w-full items-start justify-start gap-4 pb-4 leading-relaxed max-sm:mb-1 {message.routerMetadata &&
		messageInfoWidth >= messageWidth
			? 'mb-1'
			: ''}"
		data-message-id={message.id}
		data-message-role="assistant"
		role="presentation"
		onclick={() => (isTapped = !isTapped)}
		onkeydown={() => (isTapped = !isTapped)}
	>
		<MessageAvatar
			classNames="mt-5 size-3.5 flex-none select-none rounded-full shadow-lg max-sm:hidden"
			animating={isLast && loading}
		/>
		<div
			class="relative flex min-w-[60px] flex-col gap-2 break-words rounded-2xl border border-gray-100 bg-gradient-to-br from-gray-50 px-5 py-3.5 text-gray-600 prose-pre:my-2 dark:border-gray-800 dark:from-gray-800/80 dark:text-gray-300"
		>
			{#if message.files?.length}
				<div class="flex h-fit flex-wrap gap-x-5 gap-y-2">
					{#each message.files as file (file.value)}
						<UploadedFile {file} canClose={false} />
					{/each}
				</div>
			{/if}

			<div bind:this={contentEl} oncopy={handleCopy}>
				{#if isLast && loading && blocks.length === 0}
					<IconLoading classNames="loading inline ml-2 first:ml-0" />
				{/if}
				{#each blocks as block, blockIndex (block.type === "tool" ? `${block.uuid}-${blockIndex}` : `text-${blockIndex}`)}
					{@const nextBlock = blocks[blockIndex + 1]}
					{@const nextBlockHasThink =
						nextBlock?.type === "text" && THINK_BLOCK_TEST_REGEX.test(nextBlock.content)}
					{@const nextIsLinkable = nextBlock?.type === "tool" || nextBlockHasThink}
					{#if block.type === "tool"}
						<div data-exclude-from-copy class="has-[+.prose]:mb-3 [.prose+&]:mt-4">
							<ToolUpdate tool={block.updates} {loading} hasNext={nextIsLinkable} />
						</div>
					{:else if block.type === "text"}
						{#if isLast && loading && block.content.length === 0}
							<IconLoading classNames="loading inline ml-2 first:ml-0" />
						{/if}

						{#if hasClientThink}
							{@const parts = block.content.split(THINK_BLOCK_REGEX)}
							{#each parts as part, partIndex}
								{@const remainingParts = parts.slice(partIndex + 1)}
								{@const hasMoreLinkable =
									remainingParts.some((p) => p && THINK_BLOCK_TEST_REGEX.test(p)) || nextIsLinkable}
								{#if part && part.startsWith("<think>")}
									{@const isClosed = part.endsWith("</think>")}
									{@const thinkContent = part.slice(7, isClosed ? -8 : undefined)}

									<OpenReasoningResults
										content={thinkContent}
										loading={isLast && loading && !isClosed}
										hasNext={hasMoreLinkable}
									/>
								{:else if part && part.trim().length > 0}
									<div
										class="prose max-w-none dark:prose-invert max-sm:prose-sm prose-headings:font-semibold prose-h1:text-lg prose-h2:text-base prose-h3:text-base prose-pre:bg-gray-800 prose-img:my-0 prose-img:rounded-lg dark:prose-pre:bg-gray-900"
									>
										<MarkdownRenderer content={part} loading={isLast && loading} />
									</div>
								{/if}
							{/each}
						{:else}
							<div
								class="prose max-w-none dark:prose-invert max-sm:prose-sm prose-headings:font-semibold prose-h1:text-lg prose-h2:text-base prose-h3:text-base prose-pre:bg-gray-800 prose-img:my-0 prose-img:rounded-lg dark:prose-pre:bg-gray-900"
							>
								<MarkdownRenderer content={block.content} loading={isLast && loading} />
							</div>
						{/if}
					{/if}
				{/each}
			</div>
		</div>

		{#if message.routerMetadata || (!loading && message.content)}
			<div
				class="absolute -bottom-3.5 {message.routerMetadata && messageInfoWidth > messageWidth
					? 'left-1 pl-1 lg:pl-7'
					: 'right-1'} flex max-w-[calc(100dvw-40px)] items-center gap-0.5"
				bind:offsetWidth={messageInfoWidth}
			>
				{#if message.routerMetadata && (message.routerMetadata.route || message.routerMetadata.model || message.routerMetadata.provider) && (!isLast || !loading)}
					<div
						class="mr-2 flex items-center gap-1.5 truncate whitespace-nowrap text-[.65rem] text-gray-400 dark:text-gray-400 sm:text-xs"
					>
						{#if message.routerMetadata.route && message.routerMetadata.model}
							<span class="truncate rounded bg-gray-100 px-1 font-mono dark:bg-gray-800 sm:py-px">
								{message.routerMetadata.route}
							</span>
							<span class="text-gray-500">with</span>
							{#if publicConfig.isHuggingChat}
								<a
									href="/chat/settings/{message.routerMetadata.model}"
									class="flex items-center gap-1 truncate rounded bg-gray-100 px-1 font-mono hover:text-gray-500 dark:bg-gray-800 dark:hover:text-gray-300 sm:py-px"
								>
									{message.routerMetadata.model.split("/").pop()}
								</a>
							{:else}
								<span
									class="truncate rounded bg-gray-100 px-1.5 font-mono dark:bg-gray-800 sm:py-px"
								>
									{message.routerMetadata.model.split("/").pop()}
								</span>
							{/if}
						{/if}
						{#if message.routerMetadata.provider}
							{@const hubOrg = PROVIDERS_HUB_ORGS[message.routerMetadata.provider]}
							<span class="text-gray-500 max-sm:hidden">via</span>
							<a
								target="_blank"
								href="https://huggingface.co/{hubOrg}"
								class="flex items-center gap-1 truncate rounded bg-gray-100 px-1 font-mono hover:text-gray-500 dark:bg-gray-800 dark:hover:text-gray-300 max-sm:hidden sm:py-px"
							>
								<img
									src="https://huggingface.co/api/avatars/{hubOrg}"
									alt="{message.routerMetadata.provider} logo"
									class="size-2.5 flex-none rounded-sm"
									onerror={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
								/>
								{message.routerMetadata.provider}
							</a>
						{/if}
					</div>
				{/if}
				{#if !isLast || !loading}
					<div class="relative">
						<CopyToClipBoardBtn
							onClick={() => { isCopied = true; }}
							classNames="btn rounded-sm p-1 text-sm text-gray-400 hover:text-gray-500 focus:ring-0 dark:text-gray-400 dark:hover:text-gray-300"
							value={contentWithoutThink}
							iconClassNames="text-xs"
						/>
						{#if isTTSLoading}
							<div class="absolute inset-0 flex items-center justify-center pointer-events-none">
								<div class="size-3">
									<IconLoading classNames="loading" />
								</div>
							</div>
						{/if}
					</div>
					<button
						class="btn rounded-sm p-1 text-xs text-gray-400 hover:text-gray-500 focus:ring-0 dark:text-gray-400 dark:hover:text-gray-300"
						title="Read Aloud"
						type="button"
						onclick={readAloud}
						disabled={!contentWithoutThink || contentWithoutThink.trim().length === 0}
					>
						{#if isReadingAloud}
							<div class="size-3 animate-pulse">
								<CarbonTextToSpeech />
							</div>
						{:else}
							<CarbonTextToSpeech />
						{/if}
					</button>
					<button
						class="btn rounded-sm p-1 text-xs text-gray-400 hover:text-gray-500 focus:ring-0 dark:text-gray-400 dark:hover:text-gray-300"
						title="Retry"
						type="button"
						onclick={() => {
							onretry?.({ id: message.id });
						}}
					>
						<CarbonRotate360 />
					</button>
					{#if alternatives.length > 1 && editMsdgId === null}
						<Alternatives
							{message}
							{alternatives}
							{loading}
							onshowAlternateMsg={(payload) => onshowAlternateMsg?.(payload)}
						/>
					{/if}
				{/if}
			</div>
		{/if}
	</div>
{/if}
{#if message.from === "user"}
	<div
		class="group relative ml-auto flex flex-col items-end gap-4 {alternatives.length > 1 && editMsdgId === null
			? 'mb-7'
			: ''} {editMode ? 'w-full' : 'w-fit max-w-[85%]'}"
		data-message-id={message.id}
		data-message-type="user"
		role="presentation"
		onclick={() => (isTapped = !isTapped)}
		onkeydown={() => (isTapped = !isTapped)}
	>
		<div class="flex w-full flex-col gap-2">
			{#if message.files?.length}
				<div class="flex w-fit flex-wrap gap-2.5 rounded-xl bg-blue-50/50 p-3 dark:bg-blue-950/20">
					{#each message.files as file}
						<UploadedFile {file} canClose={false} />
					{/each}
				</div>
			{/if}

			<div class="flex w-full flex-row flex-nowrap">
				{#if !editMode}
					<div
						class="w-full rounded-2xl border border-blue-200/80 bg-gradient-to-br from-blue-50 to-blue-100/50 px-5 py-3.5 shadow-sm dark:border-blue-800/50 dark:from-blue-950/40 dark:to-blue-900/30 dark:shadow-blue-900/20"
					>
						<p class="whitespace-break-spaces text-wrap break-words text-gray-800 dark:text-gray-100">
							{message.content.trim()}
						</p>
					</div>
				{:else}
					<form
						class="flex w-full flex-col gap-3"
						bind:this={editFormEl}
						onsubmit={(e) => {
							e.preventDefault();
							onretry?.({ content: editContentEl?.value, id: message.id });
							editMsdgId = null;
						}}
					>
						<textarea
							class="w-full whitespace-break-spaces break-words rounded-xl border-2 border-blue-200 bg-white px-4 py-3 text-gray-800 shadow-sm transition-all focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-blue-800 dark:bg-gray-900 dark:text-gray-100 dark:focus:border-blue-600"
							rows="5"
							bind:this={editContentEl}
							value={message.content.trim()}
							onkeydown={handleKeyDown}
							required
						></textarea>
						<div class="flex w-full flex-row items-center justify-end gap-2">
							<button
								type="button"
								class="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition-all hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
								onclick={() => {
									editMsdgId = null;
								}}
							>
								Cancel
							</button>
							<button
								type="submit"
								class="rounded-lg border-0 bg-gradient-to-br px-4 py-2 text-sm font-medium shadow-md transition-all {loading
									? 'from-gray-200 to-gray-300 text-gray-400 cursor-not-allowed dark:from-gray-700 dark:to-gray-600 dark:text-gray-500'
									: 'from-blue-600 to-blue-700 text-white shadow-blue-500/30 hover:from-blue-700 hover:to-blue-800 hover:shadow-lg hover:shadow-blue-500/40 dark:from-blue-500 dark:to-blue-600 dark:shadow-blue-400/30 dark:hover:from-blue-600 dark:hover:to-blue-700'}"
								disabled={loading}
							>
								Send
							</button>
						</div>
					</form>
				{/if}
			</div>
			<div class="absolute -bottom-3.5 right-0 flex gap-2">
				{#if alternatives.length > 1 && editMsdgId === null}
					<Alternatives
						{message}
						{alternatives}
						{loading}
						onshowAlternateMsg={(payload) => onshowAlternateMsg?.(payload)}
					/>
				{/if}
				{#if (alternatives.length > 1 && editMsdgId === null) || (!loading && !editMode)}
					<button
						class="hidden cursor-pointer items-center gap-1.5 rounded-lg border border-blue-200/80 bg-white/90 px-2.5 py-1 text-xs font-medium text-blue-700 shadow-sm backdrop-blur transition-all group-hover:flex hover:flex hover:border-blue-300 hover:bg-blue-50 hover:text-blue-800 hover:shadow-md dark:border-blue-800/50 dark:bg-gray-900/90 dark:text-blue-300 dark:hover:border-blue-700 dark:hover:bg-blue-950/50 dark:hover:text-blue-200"
						title="Edit message"
						type="button"
						onclick={() => {
							if (requireAuthUser()) return;
							editMsdgId = message.id;
						}}
					>
						<CarbonPen class="size-3" />
						Edit
					</button>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	@keyframes loading {
		to {
			stroke-dashoffset: 122.9;
		}
	}
</style>
