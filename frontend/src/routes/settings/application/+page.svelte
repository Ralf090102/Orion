<script lang="ts">
	import { onMount } from "svelte";
	import CarbonSave from "~icons/carbon/save";
	import CarbonReset from "~icons/carbon/reset";

	let settings = $state({
		backendUrl: 'http://localhost:8000',
		backendWs: 'ws://localhost:8000',
		ollamaUrl: 'http://localhost:11434',
		embeddingModel: 'nomic-embed-text',
		chatModel: 'llama3.2',
		temperature: 0.7,
		maxTokens: 2048,
		topP: 0.9,
		chunkSize: 1000,
		chunkOverlap: 200,
	});

	let saved = $state(false);
	let loading = $state(false);

	onMount(async () => {
		// TODO: Load settings from backend /api/settings
		// For now, load from env/localStorage if available
		const storedBackendUrl = localStorage.getItem('backendUrl');
		if (storedBackendUrl) {
			settings.backendUrl = storedBackendUrl;
		}
	});

	async function saveSettings() {
		loading = true;
		try {
			// TODO: Send to backend /api/settings
			localStorage.setItem('backendUrl', settings.backendUrl);
			
			// Simulate API call
			await new Promise(resolve => setTimeout(resolve, 500));
			
			saved = true;
			setTimeout(() => {
				saved = false;
			}, 3000);
		} catch (err) {
			console.error('Failed to save settings:', err);
		} finally {
			loading = false;
		}
	}

	function resetToDefaults() {
		settings = {
			backendUrl: 'http://localhost:8000',
			backendWs: 'ws://localhost:8000',
			ollamaUrl: 'http://localhost:11434',
			embeddingModel: 'nomic-embed-text',
			chatModel: 'llama3.2',
			temperature: 0.7,
			maxTokens: 2048,
			topP: 0.9,
			chunkSize: 1000,
			chunkOverlap: 200,
		};
	}
</script>

<svelte:head>
	<title>Application Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<div>
		<h1 class="text-2xl font-bold">Application Settings</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Configure backend endpoints and model parameters
		</p>
	</div>

	<form onsubmit={(e) => { e.preventDefault(); saveSettings(); }} class="flex flex-col gap-6">
		<!-- Backend Configuration -->
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Backend Configuration</h2>
			
			<div class="flex flex-col gap-4">
				<div>
					<label for="backendUrl" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Backend URL
					</label>
					<input
						type="text"
						id="backendUrl"
						bind:value={settings.backendUrl}
						placeholder="http://localhost:8000"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">FastAPI backend REST endpoint</p>
				</div>

				<div>
					<label for="backendWs" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Backend WebSocket URL
					</label>
					<input
						type="text"
						id="backendWs"
						bind:value={settings.backendWs}
						placeholder="ws://localhost:8000"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">WebSocket endpoint for real-time chat</p>
				</div>

				<div>
					<label for="ollamaUrl" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Ollama URL
					</label>
					<input
						type="text"
						id="ollamaUrl"
						bind:value={settings.ollamaUrl}
						placeholder="http://localhost:11434"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Ollama API endpoint</p>
				</div>
			</div>
		</section>

		<!-- Model Configuration -->
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Model Configuration</h2>
			
			<div class="flex flex-col gap-4">
				<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
					<div>
						<label for="chatModel" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Chat Model
						</label>
						<input
							type="text"
							id="chatModel"
							bind:value={settings.chatModel}
							placeholder="llama3.2"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="embeddingModel" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Embedding Model
						</label>
						<input
							type="text"
							id="embeddingModel"
							bind:value={settings.embeddingModel}
							placeholder="nomic-embed-text"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>

				<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
					<div>
						<label for="temperature" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Temperature
						</label>
						<input
							type="number"
							id="temperature"
							bind:value={settings.temperature}
							min="0"
							max="2"
							step="0.1"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="mt-1 text-xs text-gray-500">0.0 - 2.0</p>
					</div>

					<div>
						<label for="maxTokens" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max Tokens
						</label>
						<input
							type="number"
							id="maxTokens"
							bind:value={settings.maxTokens}
							min="128"
							max="8192"
							step="128"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="topP" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Top P
						</label>
						<input
							type="number"
							id="topP"
							bind:value={settings.topP}
							min="0"
							max="1"
							step="0.1"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="mt-1 text-xs text-gray-500">0.0 - 1.0</p>
					</div>
				</div>
			</div>
		</section>

		<!-- RAG Configuration -->
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">RAG Configuration</h2>
			
			<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
				<div>
					<label for="chunkSize" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Chunk Size
					</label>
					<input
						type="number"
						id="chunkSize"
						bind:value={settings.chunkSize}
						min="100"
						max="4000"
						step="100"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Text chunk size for embeddings</p>
				</div>

				<div>
					<label for="chunkOverlap" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Chunk Overlap
					</label>
					<input
						type="number"
						id="chunkOverlap"
						bind:value={settings.chunkOverlap}
						min="0"
						max="500"
						step="50"
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Overlap between chunks</p>
				</div>
			</div>
		</section>

		<!-- Action Buttons -->
		<div class="flex items-center justify-between border-t border-gray-200 pt-6 dark:border-gray-700">
			<button
				type="button"
				onclick={resetToDefaults}
				class="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
			>
				<CarbonReset class="size-4" />
				Reset to Defaults
			</button>

			<button
				type="submit"
				disabled={loading}
				class="flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
			>
				<CarbonSave class="size-4" />
				{loading ? 'Saving...' : 'Save Settings'}
			</button>
		</div>

		{#if saved}
			<div class="rounded-lg border border-green-200 bg-green-50 p-4 text-sm text-green-800 dark:border-green-800 dark:bg-green-900/20 dark:text-green-400">
				✓ Settings saved successfully
			</div>
		{/if}
	</form>
</div>
