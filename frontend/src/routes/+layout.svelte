<script lang="ts">
	import "../styles/main.css";

	import { onDestroy, onMount, untrack } from "svelte";
	import { goto } from "$app/navigation";
	import { base } from "$app/paths";
	import { page } from "$app/state";
	import { browser } from "$app/environment";

	import { error } from "$lib/stores/errors";
	import { createSettingsStore } from "$lib/stores/settings";
	import { loading } from "$lib/stores/loading";

	import Toast from "$lib/components/Toast.svelte";
	import NavMenu from "$lib/components/NavMenu.svelte";
	import MobileNav from "$lib/components/MobileNav.svelte";
	import titleUpdate from "$lib/stores/titleUpdate";
	import ExpandNavigation from "$lib/components/ExpandNavigation.svelte";
	import { setContext } from "svelte";
	import { isAborted } from "$lib/stores/isAborted";
	import { isPro } from "$lib/stores/isPro";
	import IconShare from "$lib/components/icons/IconShare.svelte";
	import { shareModal } from "$lib/stores/shareModal";
	import ConvoModeToggle from "$lib/components/ConvoModeToggle.svelte";
	import { BACKEND_URL } from '$lib/utils/backendUrl';
	import { isTauri } from '$lib/tauri';
	import { initConversationMode, enableConversationMode } from "$lib/stores/conversationMode.svelte";

	let { data = $bindable(), children } = $props();
	
	// Backend connectivity state
	let backendConnected = $state(true);
	let backendCheckInterval: ReturnType<typeof setInterval> | null = null;

	// App update state
	let updateReady = $state(false);
	let restarting = $state(false);

	async function restartToApplyUpdate() {
		restarting = true;
		const { relaunch } = await import("@tauri-apps/plugin-process");
		await relaunch();
	}

	setContext("publicConfig", data.publicConfig);

	const publicConfig = data.publicConfig;

	let conversations = $state(data.conversations);
	$effect(() => {
		data.conversations && untrack(() => (conversations = data.conversations));
	});

	let isNavCollapsed = $state(false);

	let errorToastTimeout: ReturnType<typeof setTimeout>;
	let currentError: string | undefined = $state();

	async function onError() {
		// If a new different error comes, wait for the current error to hide first
		if ($error && currentError && $error !== currentError) {
			clearTimeout(errorToastTimeout);
			currentError = undefined;
			await new Promise((resolve) => setTimeout(resolve, 300));
		}

		currentError = $error;

		errorToastTimeout = setTimeout(() => {
			$error = undefined;
			currentError = undefined;
		}, 5000);
	}

	let canShare = $derived(
		publicConfig.isHuggingChat &&
			Boolean(page.params?.id) &&
			page.route.id?.startsWith("/conversation/")
	);

	async function deleteConversation(id: string) {
		try {
			const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${id}`, {
				method: 'DELETE',
			});

			if (!response.ok) {
				throw new Error(`Failed to delete session: ${response.statusText}`);
			}

			conversations = conversations.filter((conv) => conv.id !== id);

			if (page.params.id === id) {
				await goto(`${base}/`, { invalidateAll: true });
			}
		} catch (err) {
			console.error(err);
			$error = String(err);
		}
	}

	async function editConversationTitle(id: string, title: string) {
		try {
			const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${id}`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ title }),
			});

			if (!response.ok) {
				throw new Error(`Failed to update session title: ${response.statusText}`);
			}

			// Update local state
			conversations = conversations.map((conv) => (conv.id === id ? { ...conv, title } : conv));
		} catch (err) {
			console.error('Failed to update session title:', err);
			$error = String(err);
		}
	}

	// Voice mode has no conversation to attach to on the empty "/" screen
	// (a session only exists once something is sent). Mirrors how text chat
	// lazily creates a session on first message: jump to the most recently
	// active session if one exists, otherwise create a fresh one, then start
	// voice mode there.
	async function startVoiceModeFromEmpty() {
		try {
			let targetId: string;

			if (conversations.length > 0) {
				const topmost = [...conversations].sort(
					(a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
				)[0];
				targetId = topmost.id;
			} else {
				const response = await fetch(`${BACKEND_URL}/api/chat/sessions`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						metadata: {
							model: $settings.activeModel || "default",
							title: "New Chat",
						},
					}),
				});

				if (!response.ok) {
					throw new Error(`Failed to create session: ${response.statusText}`);
				}

				const responseData = await response.json();
				targetId = responseData.session?.session_id || responseData.session_id;

				if (!targetId) {
					throw new Error("No session ID returned from server");
				}
			}

			initConversationMode(targetId);
			enableConversationMode();
			await goto(`${base}/conversation/${targetId}`, { invalidateAll: true });
		} catch (err) {
			console.error("Failed to start voice mode:", err);
			$error = (err as Error).message || String(err);
		}
	}

	onDestroy(() => {
		clearTimeout(errorToastTimeout);
		if (backendCheckInterval) {
			clearInterval(backendCheckInterval);
		}
	});

	// Check backend connectivity
	async function checkBackendHealth() {
		if (!browser) return;
		try {
			const response = await fetch(`${BACKEND_URL}/health`, {
				method: 'GET',
				signal: AbortSignal.timeout(3000),
			});
			backendConnected = response.ok;
		} catch {
			backendConnected = false;
		}
	}

	$effect(() => {
		if ($error) onError();
	});

	$effect(() => {
		if ($titleUpdate) {
			const convIdx = conversations.findIndex(({ id }) => id === $titleUpdate?.convId);

			if (convIdx != -1) {
				conversations[convIdx].title = $titleUpdate?.title ?? conversations[convIdx].title;
			}

			$titleUpdate = null;
		}
	});

	const settings = createSettingsStore(data.settings);

	onMount(() => {
		// Global keyboard shortcut: New Chat (Ctrl/Cmd + Shift + O)
		// Registered in a synchronous onMount (not the async one below) because
		// onDestroy() must be called before any `await` — once an async onMount
		// callback resumes past its first await, Svelte's component-init window
		// has already closed and onDestroy() throws lifecycle_outside_component.
		const onKeydown = (e: KeyboardEvent) => {
			// Ignore when a modal has focus (app is inert)
			const appEl = document.getElementById("app");
			if (appEl?.hasAttribute("inert")) return;

			const oPressed = e.key?.toLowerCase() === "o";
			const metaOrCtrl = e.metaKey || e.ctrlKey;
			if (oPressed && e.shiftKey && metaOrCtrl) {
				e.preventDefault();
				isAborted.set(true);
				goto(`${base}/`, { invalidateAll: true });
			}
		};

		window.addEventListener("keydown", onKeydown, { capture: true });
		onDestroy(() => window.removeEventListener("keydown", onKeydown, { capture: true }));
	});

	onMount(async () => {
		// Check for an app update once on startup. Silent no-op outside Tauri
		// or when already up to date; installs in the background and just
		// flags a restart banner rather than force-relaunching mid-session.
		if (!isTauri()) return;
		try {
			const { check } = await import("@tauri-apps/plugin-updater");
			const update = await check();
			if (update) {
				await update.downloadAndInstall();
				updateReady = true;
			}
		} catch (err) {
			console.error("Update check failed:", err);
		}
	});

	onMount(async () => {
		// Check backend connectivity immediately and periodically
		await checkBackendHealth();
		backendCheckInterval = setInterval(checkBackendHealth, 10000); // Check every 10s

		if (page.url.searchParams.has("model")) {
			await settings
				.instantSet({
					activeModel: page.url.searchParams.get("model") ?? $settings.activeModel,
				})
				.then(async () => {
					const query = new URLSearchParams(page.url.searchParams.toString());
					query.delete("model");
					await goto(`${base}/?${query.toString()}`, {
						invalidateAll: true,
					});
				});
		}
	});

	let mobileNavTitle = $derived(
		["/models", "/privacy"].includes(page.route.id ?? "")
			? ""
			: conversations.find((conv) => conv.id === page.params.id)?.title
	);
</script>

<svelte:head>
	<title>{publicConfig.PUBLIC_APP_NAME || publicConfig.VITE_APP_NAME} - Chat with AI models</title>
	<meta name="description" content={publicConfig.PUBLIC_APP_DESCRIPTION || publicConfig.VITE_APP_DESCRIPTION} />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:site" content="@huggingface" />
	<meta name="twitter:title" content="{publicConfig.PUBLIC_APP_NAME || publicConfig.VITE_APP_NAME} - Chat with AI models" />
	<meta name="twitter:description" content={publicConfig.PUBLIC_APP_DESCRIPTION || publicConfig.VITE_APP_DESCRIPTION} />
	<meta
		name="twitter:image"
		content="{publicConfig.PUBLIC_ORIGIN || publicConfig.VITE_ORIGIN || page.url.origin}{publicConfig.assetPath}/thumbnail.png"
	/>
	<meta name="twitter:image:alt" content="{publicConfig.PUBLIC_APP_NAME || publicConfig.VITE_APP_NAME} preview" />

	<!-- use those meta tags everywhere except on special listing pages -->
	<!-- feel free to refacto if there's a better way -->
	{#if !page.url.pathname.includes("/models/")}
		<meta property="og:title" content="{publicConfig.PUBLIC_APP_NAME || publicConfig.VITE_APP_NAME} - Chat with AI models" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="{publicConfig.PUBLIC_ORIGIN || publicConfig.VITE_ORIGIN || page.url.origin}{base}" />
		<meta property="og:image" content="{publicConfig.assetPath}/thumbnail.png" />
		<meta property="og:description" content={publicConfig.PUBLIC_APP_DESCRIPTION || publicConfig.VITE_APP_DESCRIPTION} />
		<meta property="og:site_name" content={publicConfig.PUBLIC_APP_NAME || publicConfig.VITE_APP_NAME} />
		<meta property="og:locale" content="en_US" />
	{/if}
	<link rel="icon" href="{publicConfig.assetPath}/icon.svg" type="image/svg+xml" />
	{#if publicConfig.PUBLIC_ORIGIN || publicConfig.VITE_ORIGIN}
		<link
			rel="icon"
			href="{publicConfig.assetPath}/favicon.svg"
			type="image/svg+xml"
			media="(prefers-color-scheme: light)"
		/>
		<link
			rel="icon"
			href="{publicConfig.assetPath}/favicon-dark.svg"
			type="image/svg+xml"
			media="(prefers-color-scheme: dark)"
		/>
	{:else}
		<link rel="icon" href="{publicConfig.assetPath}/favicon-dev.svg" type="image/svg+xml" />
	{/if}
	<link rel="apple-touch-icon" href="{publicConfig.assetPath}/apple-touch-icon.png" />
	<link rel="manifest" href="{publicConfig.assetPath}/manifest.json" />

	{#if publicConfig.PUBLIC_PLAUSIBLE_SCRIPT_URL}
		<script async src={publicConfig.PUBLIC_PLAUSIBLE_SCRIPT_URL}></script>
	{/if}

	{#if publicConfig.PUBLIC_APPLE_APP_ID}
		<meta name="apple-itunes-app" content={`app-id=${publicConfig.PUBLIC_APPLE_APP_ID}`} />
	{/if}
</svelte:head>

<div
	class="fixed grid h-full w-screen grid-cols-1 grid-rows-[auto,1fr] overflow-hidden text-smd {!isNavCollapsed
		? 'md:grid-cols-[290px,1fr]'
		: 'md:grid-cols-[0px,1fr]'} transition-[300ms] [transition-property:grid-template-columns] dark:text-gray-300 md:grid-rows-[1fr]"
>
	<ExpandNavigation
		isCollapsed={isNavCollapsed}
		onClick={() => (isNavCollapsed = !isNavCollapsed)}
		classNames="absolute inset-y-0 z-10 my-auto {!isNavCollapsed
			? 'left-[290px]'
			: 'left-0'} *:transition-transform"
	/>

	<!-- Top-right header controls -->
	<div class="hidden md:absolute md:right-6 md:top-5 md:flex items-center gap-2">
		<!-- Conversation Mode Toggle -->
		<ConvoModeToggle onStartFromEmpty={page.params.id ? undefined : startVoiceModeFromEmpty} />
		
		<!-- Share Button -->
		{#if canShare}
			<button
				type="button"
				class="size-8 flex items-center justify-center gap-2 rounded-xl border border-gray-200 bg-white/90 text-sm font-medium text-gray-700 shadow-sm hover:bg-white/60 hover:text-gray-500 dark:border-gray-700 dark:bg-gray-800/80 dark:text-gray-200 dark:hover:bg-gray-700
					{$loading ? 'cursor-not-allowed opacity-40' : ''}"
				onclick={() => shareModal.open()}
				aria-label="Share conversation"
				disabled={$loading}
			>
				<IconShare />
			</button>
		{/if}
	</div>

	<MobileNav title={mobileNavTitle}>
		<NavMenu
			{conversations}
			user={data.user}
			ondeleteConversation={(id) => deleteConversation(id)}
			oneditConversationTitle={(payload) => editConversationTitle(payload.id, payload.title)}
		/>
	</MobileNav>
	<nav
		class="grid max-h-dvh grid-cols-1 grid-rows-[auto,1fr,auto] overflow-hidden *:w-[290px] max-md:hidden"
	>
		<NavMenu
			{conversations}
			user={data.user}
			ondeleteConversation={(id) => deleteConversation(id)}
			oneditConversationTitle={(payload) => editConversationTitle(payload.id, payload.title)}
		/>
	</nav>
	{#if currentError}
		<Toast message={currentError} />
	{/if}
	
	<!-- Backend/update status banners -->
	<div class="fixed top-0 left-0 right-0 z-50 flex flex-col">
		{#if !backendConnected}
			<div class="bg-amber-500 text-black px-4 py-2 text-center text-sm font-medium shadow-lg">
				<span class="mr-2">⚠️</span>
				Backend not connected.
				{#if isTauri()}
					Start it via Settings → Application, or run <code class="bg-amber-600/30 px-1 rounded">python -m backend.app</code> manually.
				{:else}
					Please start the backend: <code class="bg-amber-600/30 px-1 rounded">python -m backend.app</code>
				{/if}
			</div>
		{/if}
		{#if updateReady}
			<div class="bg-emerald-600 text-white px-4 py-2 text-center text-sm font-medium shadow-lg">
				<span class="mr-2">⬆️</span>
				An update has been installed.
				<button
					type="button"
					class="ml-2 underline font-semibold disabled:opacity-60"
					onclick={restartToApplyUpdate}
					disabled={restarting}
				>
					{restarting ? "Restarting…" : "Restart now"}
				</button>
			</div>
		{/if}
	</div>
	
	{@render children()}

	{#if publicConfig.PUBLIC_PLAUSIBLE_SCRIPT_URL}
		<script>
			(window.plausible =
				window.plausible ||
				function () {
					(plausible.q = plausible.q || []).push(arguments);
				}),
				(plausible.init =
					plausible.init ||
					function (i) {
						plausible.o = i || {};
					});
			plausible.init();
		</script>
	{/if}
</div>
