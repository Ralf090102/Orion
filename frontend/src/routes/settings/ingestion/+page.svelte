<script lang="ts">
	import { onMount } from "svelte";
	import CarbonUpload from "~icons/carbon/upload";
	import CarbonTrashCan from "~icons/carbon/trash-can";
	import CarbonPlay from "~icons/carbon/play";
	import CarbonStop from "~icons/carbon/stop";
	import CarbonRenew from "~icons/carbon/renew";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonWarning from "~icons/carbon/warning";
	import CarbonFolder from "~icons/carbon/folder";
	import CarbonDocument from "~icons/carbon/document";

	const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

	// Ingestion form state
	let ingestPaths = $state<string[]>([""]);
	let clearExisting = $state(false);
	let recursive = $state(true);
	let useAsync = $state(false);

	// Task management state
	let tasks = $state<Array<{
		task_id: string;
		status: string;
		path: string;
		progress: number;
		started_at: string | null;
		completed_at: string | null;
		stats: any;
		error: string | null;
	}>>([]);

	// Watchdog state
	let watchdogRunning = $state(false);
	let watchdogPaths = $state("");
	let watchdogRecursive = $state(true);
	let watchdogDebounce = $state(5);

	// UI state
	let loading = $state(false);
	let error = $state<string | null>(null);
	let success = $state<string | null>(null);
	let activeSection = $state('ingest');

	const sections = [
		{ id: 'ingest', label: 'Ingest Documents' },
		{ id: 'tasks', label: 'Task Management' },
		{ id: 'watchdog', label: 'Auto-Ingestion' },
		{ id: 'clear', label: 'Clear Knowledge Base' }
	];

	onMount(async () => {
		await loadTasks();
		await checkWatchdogStatus();
	});

	function scrollToSection(sectionId: string) {
		activeSection = sectionId;
	}

	function addPath() {
		ingestPaths = [...ingestPaths, ""];
	}

	function removePath(index: number) {
		if (ingestPaths.length > 1) {
			ingestPaths = ingestPaths.filter((_, i) => i !== index);
		}
	}

	async function loadTasks() {
		// TODO: Implement
		console.log('TODO: Load tasks from backend');
	}

	async function checkWatchdogStatus() {
		// TODO: Implement
		console.log('TODO: Check watchdog status');
	}

	async function handleIngest() {
		// TODO: Implement
		console.log('TODO: Handle document ingestion');
	}

	async function handleClearKB() {
		// TODO: Implement
		console.log('TODO: Clear knowledge base');
	}

	async function handleStartWatchdog() {
		// TODO: Implement
		console.log('TODO: Start watchdog');
	}

	async function handleStopWatchdog() {
		// TODO: Implement
		console.log('TODO: Stop watchdog');
	}

	async function deleteTask(taskId: string) {
		// TODO: Implement
		console.log('TODO: Delete task', taskId);
	}
</script>

<svelte:head>
	<title>Document Ingestion - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<div>
		<h1 class="text-2xl font-bold">Document Ingestion</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Manage document ingestion and knowledge base
		</p>
	</div>

	<!-- Info Banner -->
	<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
		<div class="flex items-start gap-3">
			<CarbonDocument class="size-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
			<div class="flex-1">
				<h3 class="font-semibold text-blue-900 dark:text-blue-200 text-sm">
					About Document Ingestion
				</h3>
				<p class="text-sm text-blue-800 dark:text-blue-300 mt-1">
					Ingest documents to make them available for RAG-based conversations. Supports various formats including PDF, TXT, MD, DOCX, and more. Use async ingestion for large datasets.
				</p>
			</div>
		</div>
	</div>

	<!-- Error Message -->
	{#if error}
		<div class="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
			<div class="flex items-start gap-3">
				<CarbonWarning class="size-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<h3 class="font-semibold text-red-900 dark:text-red-200 text-sm">Error</h3>
					<p class="text-sm text-red-800 dark:text-red-300 mt-1">{error}</p>
				</div>
				<button
					onclick={() => error = null}
					class="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200"
				>
					×
				</button>
			</div>
		</div>
	{/if}

	<!-- Success Message -->
	{#if success}
		<div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
			<div class="flex items-start gap-3">
				<CarbonCheckmark class="size-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<p class="text-sm text-green-800 dark:text-green-300">{success}</p>
				</div>
				<button
					onclick={() => success = null}
					class="text-green-600 hover:text-green-800 dark:text-green-400 dark:hover:text-green-200"
				>
					×
				</button>
			</div>
		</div>
	{/if}

	<!-- Navigation Tabs -->
	<div class="sticky top-0 z-10 -mx-5 sm:-mx-8 px-5 sm:px-8 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 py-3">
		<div class="flex gap-2 overflow-x-auto scrollbar-custom pb-1">
			{#each sections as section}
				<button
					type="button"
					onclick={() => scrollToSection(section.id)}
					class="px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors {activeSection === section.id 
						? 'bg-blue-600 text-white dark:bg-blue-500' 
						: 'bg-white text-gray-700 hover:bg-gray-100 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'}"
				>
					{section.label}
				</button>
			{/each}
		</div>
	</div>

	<div class="flex flex-col gap-6">
		<!-- Ingest Documents Section -->
		{#if activeSection === 'ingest'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Ingest Documents</h2>
			
			<div class="flex flex-col gap-4">
				<div>
					<div class="flex items-center justify-between mb-2">
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
							Path(s) to File or Directory
						</label>
						<button
							type="button"
							onclick={addPath}
							class="flex items-center gap-1 rounded-lg bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
						>
							<span class="text-lg leading-none">+</span>
							Add Path
						</button>
					</div>
					
					<div class="space-y-2">
						{#each ingestPaths as path, index}
							<div class="flex gap-2">
								<input
									type="text"
									bind:value={ingestPaths[index]}
									placeholder="D:/Documents/Books or ./data/knowledge_base"
									class="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
								/>
								<button
									type="button"
									class="rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
								>
									<CarbonFolder class="size-5" />
								</button>
								{#if ingestPaths.length > 1}
									<button
										type="button"
										onclick={() => removePath(index)}
										class="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-red-600 hover:bg-red-100 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30"
									>
										<CarbonTrashCan class="size-5" />
									</button>
								{/if}
							</div>
						{/each}
					</div>
					<p class="mt-2 text-xs text-gray-500">Enter absolute or relative paths to documents. Add multiple paths to ingest from different locations.</p>
				</div>

				<div class="flex flex-wrap gap-4">
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={clearExisting}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Clear Existing Knowledge Base</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={recursive}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Recursive (Include Subdirectories)</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={useAsync}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Async Mode (For Large Datasets)</span>
					</label>
				</div>

				{#if clearExisting}
					<div class="rounded-lg border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-900/20">
						<div class="flex items-start gap-2">
							<CarbonWarning class="size-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
							<p class="text-xs text-amber-800 dark:text-amber-300">
								Warning: This will permanently delete all existing documents and embeddings before ingestion.
							</p>
						</div>
					</div>
				{/if}

				<div class="flex items-center justify-end gap-3 pt-2">
					<button
						type="button"
						onclick={handleIngest}
						disabled={loading || !ingestPaths.some(p => p.trim())}
						class="flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
					>
						<CarbonUpload class="size-4" />
						{loading ? 'Processing...' : 'Start Ingestion'}
					</button>
				</div>
			</div>
		</section>
		{/if}

		<!-- Task Management Section -->
		{#if activeSection === 'tasks'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100">Ingestion Tasks</h2>
				<button
					type="button"
					onclick={loadTasks}
					class="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
				>
					<CarbonRenew class="size-4" />
					Refresh
				</button>
			</div>
			
			{#if tasks.length === 0}
				<div class="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center dark:border-gray-700 dark:bg-gray-800/50">
					<CarbonDocument class="mx-auto size-12 text-gray-400 dark:text-gray-500" />
					<p class="mt-2 text-sm text-gray-600 dark:text-gray-400">No ingestion tasks yet</p>
					<p class="mt-1 text-xs text-gray-500 dark:text-gray-500">Tasks will appear here after you start an async ingestion</p>
				</div>
			{:else}
				<div class="space-y-3">
					{#each tasks as task}
						<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
							<div class="flex items-start justify-between">
								<div class="flex-1">
									<div class="flex items-center gap-2">
										<span class="text-sm font-medium text-gray-900 dark:text-gray-100">
											{task.path}
										</span>
										<span class="rounded-full px-2 py-0.5 text-xs font-medium {
											task.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
											task.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
											task.status === 'running' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
											'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
										}">
											{task.status}
										</span>
									</div>
									
									{#if task.status === 'running'}
										<div class="mt-2">
											<div class="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
												<span>Progress</span>
												<span>{task.progress.toFixed(1)}%</span>
											</div>
											<div class="h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
												<div 
													class="h-full bg-blue-600 dark:bg-blue-500 transition-all duration-300"
													style="width: {task.progress}%"
												></div>
											</div>
										</div>
									{/if}

									{#if task.stats}
										<div class="mt-2 grid grid-cols-3 gap-3 text-xs">
											<div>
												<span class="text-gray-500 dark:text-gray-400">Files:</span>
												<span class="ml-1 font-medium text-gray-900 dark:text-gray-100">
													{task.stats.successful_files}/{task.stats.total_files}
												</span>
											</div>
											<div>
												<span class="text-gray-500 dark:text-gray-400">Chunks:</span>
												<span class="ml-1 font-medium text-gray-900 dark:text-gray-100">
													{task.stats.total_chunks}
												</span>
											</div>
											<div>
												<span class="text-gray-500 dark:text-gray-400">Time:</span>
												<span class="ml-1 font-medium text-gray-900 dark:text-gray-100">
													{task.stats.processing_time?.toFixed(2)}s
												</span>
											</div>
										</div>
									{/if}

									{#if task.error}
										<p class="mt-2 text-xs text-red-600 dark:text-red-400">{task.error}</p>
									{/if}
								</div>

								<button
									type="button"
									onclick={() => deleteTask(task.task_id)}
									class="ml-4 text-gray-400 hover:text-red-600 dark:text-gray-500 dark:hover:text-red-400"
								>
									<CarbonTrashCan class="size-4" />
								</button>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</section>
		{/if}

		<!-- Auto-Ingestion (Watchdog) Section -->
		{#if activeSection === 'watchdog'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Auto-Ingestion (Watchdog)</h2>
			
			<div class="flex flex-col gap-4">
				<div class="rounded-lg border border-blue-200 bg-blue-50 p-3 dark:border-blue-800 dark:bg-blue-900/20">
					<p class="text-xs text-blue-800 dark:text-blue-300">
						The watchdog monitors directories and automatically ingests new or modified files into the knowledge base.
					</p>
				</div>

				<div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50">
					<div class="flex items-center gap-2">
						<div class="size-3 rounded-full {watchdogRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}"></div>
						<span class="text-sm font-medium text-gray-900 dark:text-gray-100">
							Status: {watchdogRunning ? 'Running' : 'Stopped'}
						</span>
					</div>
					<button
						type="button"
						onclick={watchdogRunning ? handleStopWatchdog : handleStartWatchdog}
						disabled={loading}
						class="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium {
							watchdogRunning 
								? 'bg-red-600 text-white hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600' 
								: 'bg-green-600 text-white hover:bg-green-700 dark:bg-green-500 dark:hover:bg-green-600'
						} disabled:opacity-50"
					>
						{#if watchdogRunning}
							<CarbonStop class="size-4" />
							Stop Watchdog
						{:else}
							<CarbonPlay class="size-4" />
							Start Watchdog
						{/if}
					</button>
				</div>

				<div>
					<label for="watchdog-paths" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Watch Paths (one per line)
					</label>
					<textarea
						id="watchdog-paths"
						bind:value={watchdogPaths}
						rows="4"
						placeholder="D:/Documents/Books&#10;./data/incoming&#10;C:/Projects/docs"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					></textarea>
					<p class="mt-1 text-xs text-gray-500">Enter one directory path per line to watch for changes</p>
				</div>

				<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
					<div>
						<label for="watchdog-debounce" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Debounce (seconds)
						</label>
						<input
							type="number"
							id="watchdog-debounce"
							bind:value={watchdogDebounce}
							min="1"
							max="60"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="mt-1 text-xs text-gray-500">Wait time before ingesting after file change</p>
					</div>

					<div class="flex items-center pt-6">
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={watchdogRecursive}
								class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
							/>
							<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Watch Subdirectories</span>
						</label>
					</div>
				</div>
			</div>
		</section>
		{/if}

		<!-- Clear Knowledge Base Section -->
		{#if activeSection === 'clear'}
		<section class="rounded-xl border border-red-200 bg-red-50 p-6 dark:border-red-800 dark:bg-red-900/20">
			<h2 class="mb-4 text-lg font-semibold text-red-900 dark:text-red-200">Clear Knowledge Base</h2>
			
			<div class="flex flex-col gap-4">
				<div class="rounded-lg border border-red-300 bg-red-100 p-4 dark:border-red-700 dark:bg-red-900/30">
					<div class="flex items-start gap-3">
						<CarbonWarning class="size-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
						<div class="flex-1">
							<h3 class="font-semibold text-red-900 dark:text-red-200 text-sm">Danger Zone</h3>
							<p class="text-sm text-red-800 dark:text-red-300 mt-1">
								This action will permanently delete all documents and embeddings from the knowledge base. This operation cannot be undone!
							</p>
						</div>
					</div>
				</div>

				<div class="space-y-3">
					<p class="text-sm text-red-800 dark:text-red-300">
						Before clearing the knowledge base, consider:
					</p>
					<ul class="list-disc list-inside space-y-1 text-sm text-red-700 dark:text-red-400 ml-2">
						<li>All ingested documents will be removed</li>
						<li>All generated embeddings will be deleted</li>
						<li>RAG-based conversations will not have access to any documents</li>
						<li>You will need to re-ingest documents to restore functionality</li>
					</ul>
				</div>

				<div class="flex items-center justify-end gap-3 pt-2">
					<button
						type="button"
						onclick={handleClearKB}
						disabled={loading}
						class="flex items-center gap-2 rounded-lg bg-red-600 px-6 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50 dark:bg-red-500 dark:hover:bg-red-600"
					>
						<CarbonTrashCan class="size-4" />
						{loading ? 'Clearing...' : 'Clear Knowledge Base'}
					</button>
				</div>
			</div>
		</section>
		{/if}
	</div>
</div>
