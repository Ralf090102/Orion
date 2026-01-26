<script lang="ts">
	import { onMount } from "svelte";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonClose from "~icons/carbon/close";

	let models = $state<Array<{
		id: string;
		name: string;
		size: string;
		active: boolean;
		description?: string;
	}>>([]);

	let loading = $state(true);

	onMount(async () => {
		// TODO: Fetch from backend /api/settings/models
		// Mock data for now
		models = [
			{
				id: "llama3.2",
				name: "Llama 3.2",
				size: "3.2B",
				active: true,
				description: "Latest Llama model, great for general tasks"
			},
			{
				id: "mistral",
				name: "Mistral",
				size: "7B",
				active: true,
				description: "Fast and efficient for most use cases"
			},
			{
				id: "codellama",
				name: "Code Llama",
				size: "7B",
				active: false,
				description: "Specialized for code generation and understanding"
			},
		];
		loading = false;
	});

	function toggleModel(modelId: string) {
		const model = models.find(m => m.id === modelId);
		if (model) {
			model.active = !model.active;
			models = [...models]; // Trigger reactivity
			// TODO: Send update to backend
		}
	}
</script>

<svelte:head>
	<title>Models - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<div>
		<h1 class="text-2xl font-bold">Ollama Models</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Manage which models are available for chat. Active models can be selected in conversations.
		</p>
	</div>

	{#if loading}
		<div class="flex items-center justify-center py-12">
			<div class="text-gray-500">Loading models...</div>
		</div>
	{:else if models.length === 0}
		<div class="flex flex-col items-center justify-center py-12 gap-4">
			<p class="text-gray-500">No models found</p>
			<p class="text-sm text-gray-400">Make sure Ollama is running and has models installed</p>
		</div>
	{:else}
		<div class="flex flex-col gap-4">
			{#each models as model (model.id)}
				<div
					class="flex items-center justify-between rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800"
				>
					<div class="flex flex-col gap-1">
						<div class="flex items-center gap-2">
							<h3 class="font-semibold text-gray-900 dark:text-gray-100">{model.name}</h3>
							<span class="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600 dark:bg-gray-700 dark:text-gray-400">
								{model.size}
							</span>
						</div>
						{#if model.description}
							<p class="text-sm text-gray-600 dark:text-gray-400">{model.description}</p>
						{/if}
						<p class="text-xs text-gray-500 dark:text-gray-500">Model ID: {model.id}</p>
					</div>

					<button
						onclick={() => toggleModel(model.id)}
						class="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors
							{model.active
								? 'border-green-200 bg-green-50 text-green-700 hover:bg-green-100 dark:border-green-800 dark:bg-green-900/20 dark:text-green-400 dark:hover:bg-green-900/30'
								: 'border-gray-300 bg-gray-50 text-gray-600 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600'}"
					>
						{#if model.active}
							<CarbonCheckmark class="size-4" />
							Active
						{:else}
							<CarbonClose class="size-4" />
							Inactive
						{/if}
					</button>
				</div>
			{/each}
		</div>
	{/if}

	<div class="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
		<h3 class="font-semibold text-blue-900 dark:text-blue-300">Add New Models</h3>
		<p class="mt-1 text-sm text-blue-800 dark:text-blue-400">
			To add new models, use the Ollama CLI: <code class="rounded bg-blue-100 px-1 py-0.5 dark:bg-blue-900/40">ollama pull &lt;model-name&gt;</code>
		</p>
		<p class="mt-2 text-sm text-blue-800 dark:text-blue-400">
			After pulling a new model, refresh this page to see it in the list.
		</p>
	</div>
</div>
