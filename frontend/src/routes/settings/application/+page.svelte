<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import CarbonSettings from "~icons/carbon/settings";
	import CarbonChevronRight from "~icons/carbon/chevron-right";
	import CarbonDocumentTasks from "~icons/carbon/document-tasks";
	import CarbonUpload from "~icons/carbon/upload";
	import CarbonVolumeUp from "~icons/carbon/volume-up";
	import CarbonMicrophone from "~icons/carbon/microphone";
	import CarbonCheckmarkFilled from "~icons/carbon/checkmark-filled";
	import CarbonWarningFilled from "~icons/carbon/warning-filled";
	import CarbonStopFilled from "~icons/carbon/stop-filled";
	import CarbonPlay from "~icons/carbon/play";
	import CarbonReset from "~icons/carbon/reset";

	import { isTauri, pollBackendStatus, startBackend, stopBackend, restartBackend } from '$lib/tauri';

	let backendStatus: string = 'checking';
	let isDesktopMode = false;
	let pollCleanup: (() => void) | null = null;
	let isProcessing = false;

	onMount(() => {
		isDesktopMode = isTauri();
		
		if (isDesktopMode) {
			// Start polling backend status
			pollCleanup = pollBackendStatus((status) => {
				backendStatus = status;
			}, 5000); // Check every 5 seconds
		}
	});

	onDestroy(() => {
		if (pollCleanup) {
			pollCleanup();
		}
	});

	async function handleStartBackend() {
		isProcessing = true;
		try {
			await startBackend();
			backendStatus = 'starting';
		} catch (error) {
			console.error('Failed to start backend:', error);
		} finally {
			isProcessing = false;
		}
	}

	async function handleStopBackend() {
		isProcessing = true;
		try {
			await stopBackend();
			backendStatus = 'stopped';
		} catch (error) {
			console.error('Failed to stop backend:', error);
		} finally {
			isProcessing = false;
		}
	}

	async function handleRestartBackend() {
		isProcessing = true;
		try {
			await restartBackend();
			backendStatus = 'restarting';
		} catch (error) {
			console.error('Failed to restart backend:', error);
		} finally {
			isProcessing = false;
		}
	}

	const settingsCategories = [
		{
			id: 'rag',
			title: 'RAG Pipeline',
			description: 'Configure retrieval, embedding, chunking, and generation settings',
			icon: CarbonDocumentTasks,
			path: '/settings/rag',
			enabled: true
		},
		{
			id: 'ingestion',
			title: 'Document Ingestion',
			description: 'Ingest documents, manage knowledge base, and configure auto-ingestion',
			icon: CarbonUpload,
			path: '/settings/ingestion',
			enabled: true
		},
		{
			id: 'speech',
			title: 'Speech & Audio',
			description: 'Configure speech-to-text (STT) and text-to-speech (TTS) settings',
			icon: CarbonVolumeUp,
			path: '/settings/speech',
			enabled: true
		},
		{
			id: 'conversation',
			title: 'Conversation Mode',
			description: 'Configure voice conversation settings: VAD, auto-TTS, input modes',
			icon: CarbonMicrophone,
			path: '/settings/conversation',
			enabled: true
		},
		// Future settings categories will be added here
		// {
		//   id: 'backend',
		//   title: 'Backend Configuration',
		//   description: 'Configure backend endpoints and connections',
		//   icon: CarbonServer,
		//   path: '/settings/backend',
		//   enabled: false
		// }
	];
</script>

<svelte:head>
	<title>Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<div>
		<h1 class="text-2xl font-bold">Settings</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Configure your Orion application
		</p>
	</div>

	<!-- Backend Status (Desktop Only) -->
	{#if isDesktopMode}
		<div class="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-4">
					<div class="flex items-center gap-2">
						{#if backendStatus === 'running'}
							<CarbonCheckmarkFilled class="size-6 text-green-500" />
							<div>
								<h3 class="font-semibold text-gray-900 dark:text-gray-100">Backend Running</h3>
								<p class="text-sm text-gray-600 dark:text-gray-400">Python backend is active</p>
							</div>
						{:else if backendStatus === 'stopped'}
							<CarbonStopFilled class="size-6 text-red-500" />
							<div>
								<h3 class="font-semibold text-gray-900 dark:text-gray-100">Backend Stopped</h3>
								<p class="text-sm text-gray-600 dark:text-gray-400">Backend is not running</p>
							</div>
						{:else if backendStatus === 'checking' || backendStatus === 'starting' || backendStatus === 'restarting'}
							<div class="size-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"></div>
							<div>
								<h3 class="font-semibold text-gray-900 dark:text-gray-100">
									{backendStatus === 'checking' ? 'Checking Status' : backendStatus === 'starting' ? 'Starting Backend' : 'Restarting Backend'}
								</h3>
								<p class="text-sm text-gray-600 dark:text-gray-400">Please wait...</p>
							</div>
						{:else}
							<CarbonWarningFilled class="size-6 text-yellow-500" />
							<div>
								<h3 class="font-semibold text-gray-900 dark:text-gray-100">Backend Error</h3>
								<p class="text-sm text-gray-600 dark:text-gray-400">Status: {backendStatus}</p>
							</div>
						{/if}
					</div>
				</div>

				<!-- Control Buttons -->
				<div class="flex items-center gap-2">
					{#if backendStatus === 'running'}
						<button
							on:click={handleRestartBackend}
							disabled={isProcessing}
							class="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
						>
							<CarbonReset class="size-4" />
							Restart
						</button>
						<button
							on:click={handleStopBackend}
							disabled={isProcessing}
							class="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
						>
							<CarbonStopFilled class="size-4" />
							Stop
						</button>
					{:else if backendStatus === 'stopped'}
						<button
							on:click={handleStartBackend}
							disabled={isProcessing}
							class="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
						>
							<CarbonPlay class="size-4" />
							Start
						</button>
					{/if}
				</div>
			</div>
		</div>
	{/if}

	<div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
		{#each settingsCategories as category}
			{#if category.enabled}
				<a
					href={category.path}
					class="group rounded-xl border border-gray-200 bg-white p-6 transition-all hover:border-blue-500 hover:shadow-md dark:border-gray-700 dark:bg-gray-800 dark:hover:border-blue-500"
				>
					<div class="flex items-start justify-between">
						<div class="flex items-start gap-4">
							<div class="rounded-lg bg-blue-100 p-3 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
								<svelte:component this={category.icon} class="size-6" />
							</div>
							<div class="flex-1">
								<h3 class="font-semibold text-gray-900 dark:text-gray-100">
									{category.title}
								</h3>
								<p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
									{category.description}
								</p>
							</div>
						</div>
						<CarbonChevronRight class="size-5 text-gray-400 transition-transform group-hover:translate-x-1 dark:text-gray-500" />
					</div>
				</a>
			{:else}
				<div class="rounded-xl border border-gray-200 bg-gray-50 p-6 opacity-60 dark:border-gray-700 dark:bg-gray-800/50">
					<div class="flex items-start gap-4">
						<div class="rounded-lg bg-gray-200 p-3 text-gray-400 dark:bg-gray-700 dark:text-gray-500">
							<svelte:component this={category.icon} class="size-6" />
						</div>
						<div class="flex-1">
							<h3 class="font-semibold text-gray-500 dark:text-gray-400">
								{category.title}
							</h3>
							<p class="mt-1 text-sm text-gray-500 dark:text-gray-500">
								{category.description}
							</p>
							<p class="mt-2 text-xs text-gray-400 dark:text-gray-500">
								Coming soon
							</p>
						</div>
					</div>
				</div>
			{/if}
		{/each}
	</div>

	<!-- Info Box -->
	<div class="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
		<div class="flex items-start gap-3">
			<CarbonSettings class="size-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
			<div class="flex-1">
				<h3 class="font-semibold text-blue-900 dark:text-blue-200 text-sm">
					Settings Organization
				</h3>
				<p class="text-sm text-blue-800 dark:text-blue-300 mt-1">
					Settings are organized by feature category. Click on a category to configure its settings. More categories will be added as new features are developed.
				</p>
			</div>
		</div>
	</div>
</div>
