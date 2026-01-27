<script lang="ts">
	import { onMount } from "svelte";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonClose from "~icons/carbon/close";
	import CarbonOverflowMenuVertical from "~icons/carbon/overflow-menu-vertical";
	import { env as publicEnv } from "$env/dynamic/public";
	import { useSettingsStore } from "$lib/stores/settings.js";
	import ModelParametersModal from "$lib/components/ModelParametersModal.svelte";

	const settings = useSettingsStore();
	const BACKEND_URL = publicEnv.PUBLIC_BACKEND_URL || "http://localhost:8000";

	let models = $state<Array<{
		id: string;
		name: string;
		size: string;
		size_bytes: number;
		modified: string;
		active: boolean;
		details?: {
			format?: string;
			family?: string;
			parameter_size?: string;
		};
	}>>([]);

	let loading = $state(true);
	let error = $state<string | null>(null);
	let showParametersModal = $state(false);
	let selectedModelForSettings = $state<string>("");

	// Models to hide (embedding models, not for generation)
	const HIDDEN_MODELS = ["nomic-embed-text", "nomic-embed"];

	onMount(async () => {
		await fetchModels();
	});

	async function fetchModels() {
		loading = true;
		error = null;
		
		try {
			const response = await fetch(`${BACKEND_URL}/api/models`);
			
			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || "Failed to fetch models");
			}

			const data = await response.json();
			
			// Filter out embedding models and map to UI format
			const allModels = data.models || [];
			const filteredModels = allModels.filter((model: any) => {
				const modelName = model.name.toLowerCase();
				return !HIDDEN_MODELS.some(hidden => modelName.includes(hidden));
			});

			// Get current active model from backend
			const activeModelName = data.current_model;

			models = filteredModels.map((model: any) => ({
				id: model.id,
				name: model.name,
				size: model.size,
				size_bytes: model.size_bytes,
				modified: model.modified,
				active: model.name === activeModelName,
				details: model.details,
			}));

			// If no model is active but we have models, activate the first one
			if (models.length > 0 && !models.some(m => m.active)) {
				models[0].active = true;
				await setActiveModel(models[0].name);
			}
		} catch (err) {
			console.error("Failed to fetch models:", err);
			error = err instanceof Error ? err.message : "Failed to fetch models";
		} finally {
			loading = false;
		}
	}

	async function setActiveModel(modelName: string) {
		try {
			// Update backend configuration
			const response = await fetch(`${BACKEND_URL}/api/models/config`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ model: modelName }),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || "Failed to update model");
			}

			// Deactivate all models
			models = models.map(m => ({ ...m, active: false }));
			
			// Activate the selected model
			const selectedModel = models.find(m => m.name === modelName);
			if (selectedModel) {
				selectedModel.active = true;
				models = [...models]; // Trigger reactivity
			}

			// Also update settings store for consistency
			settings.set({ activeModel: modelName });
			
			console.log(`✅ Active model updated to: ${modelName}`);
		} catch (err) {
			console.error("Failed to set active model:", err);
			error = err instanceof Error ? err.message : "Failed to set active model";
			
			// Revert UI changes on error
			await fetchModels();
		}
	}

	function openParametersModal(modelName: string) {
		selectedModelForSettings = modelName;
		showParametersModal = true;
	}
</script>

<svelte:head>
	<title>Models - Orion</title>
</svelte:head>

{#if showParametersModal}
	<ModelParametersModal 
		bind:open={showParametersModal}
		onClose={() => showParametersModal = false}
		modelName={selectedModelForSettings}
	/>
{/if}

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
	{:else if error}
		<div class="flex flex-col items-center justify-center py-12 gap-4">
			<p class="text-red-500 font-semibold">Error loading models</p>
			<p class="text-sm text-gray-400">{error}</p>
			<button
				onclick={fetchModels}
				class="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
			>
				Retry
			</button>
		</div>
	{:else if models.length === 0}
		<div class="flex flex-col items-center justify-center py-12 gap-4">
			<p class="text-gray-500">No models found</p>
			<p class="text-sm text-gray-400">Make sure Ollama is running and has models installed</p>
			<button
				onclick={fetchModels}
				class="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
			>
				Refresh
			</button>
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
							{#if model.details?.parameter_size}
								<span class="rounded-full bg-purple-100 px-2 py-0.5 text-xs text-purple-600 dark:bg-purple-900/40 dark:text-purple-400">
									{model.details.parameter_size}
								</span>
							{/if}
						</div>
						{#if model.details?.family}
							<p class="text-sm text-gray-600 dark:text-gray-400">
								Family: {model.details.family}
								{#if model.details?.format}
									• Format: {model.details.format}
								{/if}
							</p>
						{/if}
						<p class="text-xs text-gray-500 dark:text-gray-500">Model ID: {model.id}</p>
					</div>

					<div class="flex items-center gap-2">
						<button
							onclick={() => setActiveModel(model.name)}
							disabled={model.active}
							class="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors
								{model.active
									? 'border-green-200 bg-green-50 text-green-700 cursor-default dark:border-green-800 dark:bg-green-900/20 dark:text-green-400'
									: 'border-gray-300 bg-gray-50 text-gray-600 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600 cursor-pointer'}"
						>
							{#if model.active}
								<CarbonCheckmark class="size-4" />
								Active
							{:else}
								<CarbonClose class="size-4" />
								Inactive
							{/if}
						</button>

						<button
							onclick={() => openParametersModal(model.name)}
							class="rounded-lg border border-gray-300 bg-white p-2 text-gray-600 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700 transition-colors"
							title="Model settings"
						>
							<CarbonOverflowMenuVertical class="size-5" />
						</button>
					</div>
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
