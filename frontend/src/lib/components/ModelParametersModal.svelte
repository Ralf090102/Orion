<script lang="ts">
	import { env as publicEnv } from "$env/dynamic/public";
	import Modal from "$lib/components/Modal.svelte";
	import CarbonClose from "~icons/carbon/close";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonReset from "~icons/carbon/reset";

	interface Props {
		open: boolean;
		onClose: () => void;
		modelName: string;
	}

	let { open = $bindable(false), onClose, modelName }: Props = $props();

	const BACKEND_URL = publicEnv.PUBLIC_BACKEND_URL || "http://localhost:8000";

	// Default values
	const DEFAULT_TEMPERATURE = 0.7;
	const DEFAULT_TOP_P = 0.9;
	const DEFAULT_MAX_TOKENS = null;
	const DEFAULT_TIMEOUT = 90;
	const DEFAULT_SYSTEM_PROMPT = "You are Orion, a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and cite sources when appropriate.";

	// Model parameters
	let temperature = $state(DEFAULT_TEMPERATURE);
	let top_p = $state(DEFAULT_TOP_P);
	let max_tokens = $state<number | null>(DEFAULT_MAX_TOKENS);
	let timeout = $state(DEFAULT_TIMEOUT);
	let system_prompt = $state(DEFAULT_SYSTEM_PROMPT);

	let loading = $state(false);
	let saving = $state(false);
	let error = $state<string | null>(null);
	let successMessage = $state<string | null>(null);

	// Load current configuration when modal opens
	$effect(() => {
		if (open) {
			loadConfiguration();
		}
	});

	async function loadConfiguration() {
		loading = true;
		error = null;

		try {
			const response = await fetch(`${BACKEND_URL}/api/models/config`);

			if (!response.ok) {
				throw new Error("Failed to load configuration");
			}

			const config = await response.json();

			temperature = config.temperature;
			top_p = config.top_p;
			max_tokens = config.max_tokens;
			timeout = config.timeout;
			system_prompt = config.system_prompt;
		} catch (err) {
			console.error("Failed to load configuration:", err);
			error = err instanceof Error ? err.message : "Failed to load configuration";
		} finally {
			loading = false;
		}
	}

	async function saveConfiguration() {
		saving = true;
		error = null;
		successMessage = null;

		try {
			const response = await fetch(`${BACKEND_URL}/api/models/parameters`, {
				method: "PATCH",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					temperature,
					top_p,
					max_tokens,
					timeout,
					system_prompt,
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || "Failed to save configuration");
			}

			successMessage = "Configuration saved successfully!";
			setTimeout(() => {
				successMessage = null;
			}, 3000);
		} catch (err) {
			console.error("Failed to save configuration:", err);
			error = err instanceof Error ? err.message : "Failed to save configuration";
		} finally {
			saving = false;
		}
	}

	function resetToDefaults() {
		temperature = DEFAULT_TEMPERATURE;
		top_p = DEFAULT_TOP_P;
		max_tokens = DEFAULT_MAX_TOKENS;
		timeout = DEFAULT_TIMEOUT;
		system_prompt = DEFAULT_SYSTEM_PROMPT;
		successMessage = null;
		error = null;
	}

	function handleClose() {
		successMessage = null;
		error = null;
		onClose();
	}
</script>

<Modal open={open} onclose={handleClose} width="w-[900px]">
	<div class="flex flex-col gap-4 pl-6 pr-0 py-6">
		<!-- Header -->
		<div class="border-b border-gray-200 pb-4 dark:border-gray-700 pr-6">
			<h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">Model Parameters</h2>
			<p class="text-sm text-gray-500 dark:text-gray-400 mt-1">{modelName}</p>
		</div>

		<!-- Content -->
		{#if loading}
			<div class="flex items-center justify-center py-12">
				<div class="text-gray-500">Loading configuration...</div>
			</div>
		{:else}
			<div class="scrollbar-custom flex flex-col gap-4 max-h-[60vh] overflow-y-auto pr-6">
				<!-- Temperature -->
				<div class="flex flex-col gap-2">
					<label for="temperature" class="text-sm font-medium text-gray-700 dark:text-gray-300">
						Temperature
					</label>
					<div class="flex items-center gap-4">
						<input
							id="temperature"
							type="range"
							min="0"
							max="2"
							step="0.01"
							bind:value={temperature}
							class="flex-1 accent-blue-600"
						/>
						<input
							type="number"
							min="0"
							max="2"
							step="0.01"
							bind:value={temperature}
							class="w-20 rounded-lg border border-gray-300 bg-white px-3 py-1 text-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
						/>
					</div>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Controls randomness. Lower is more focused, higher is more creative. (0.0 - 2.0)
					</p>
				</div>

				<!-- Top P -->
				<div class="flex flex-col gap-2">
					<label for="top_p" class="text-sm font-medium text-gray-700 dark:text-gray-300">
						Top P
					</label>
					<div class="flex items-center gap-4">
						<input
							id="top_p"
							type="range"
							min="0"
							max="1"
							step="0.01"
							bind:value={top_p}
							class="flex-1 accent-blue-600"
						/>
						<input
							type="number"
							min="0"
							max="1"
							step="0.01"
							bind:value={top_p}
							class="w-20 rounded-lg border border-gray-300 bg-white px-3 py-1 text-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
						/>
					</div>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Nucleus sampling. Controls diversity of responses. (0.0 - 1.0)
					</p>
				</div>

				<!-- Max Tokens -->
				<div class="flex flex-col gap-2">
					<label for="max_tokens" class="text-sm font-medium text-gray-700 dark:text-gray-300">
						Max Tokens
					</label>
					<input
						id="max_tokens"
						type="number"
						min="1"
						bind:value={max_tokens}
						placeholder="Unlimited"
						class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
					/>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Maximum tokens to generate. Leave empty for unlimited.
					</p>
				</div>

				<!-- Timeout -->
				<div class="flex flex-col gap-2">
					<label for="timeout" class="text-sm font-medium text-gray-700 dark:text-gray-300">
						Timeout (seconds)
					</label>
					<input
						id="timeout"
						type="number"
						min="1"
						bind:value={timeout}
						class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
					/>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Request timeout in seconds.
					</p>
				</div>

				<!-- System Prompt -->
				<div class="flex flex-col gap-2">
					<label for="system_prompt" class="text-sm font-medium text-gray-700 dark:text-gray-300">
						System Prompt
					</label>
					<textarea
						id="system_prompt"
						bind:value={system_prompt}
						rows="6"
						class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
						placeholder="Enter system prompt..."
					></textarea>
					<p class="text-xs text-gray-500 dark:text-gray-400">
						Instructions that guide the model's behavior and personality.
					</p>
				</div>
			</div>

			<!-- Messages -->
			{#if error}
				<div class="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-400 mr-6">
					{error}
				</div>
			{/if}

			{#if successMessage}
				<div class="rounded-lg bg-green-50 p-3 text-sm text-green-700 dark:bg-green-900/20 dark:text-green-400 flex items-center gap-2 mr-6">
					<CarbonCheckmark class="size-4" />
					{successMessage}
				</div>
			{/if}

			<!-- Actions -->
			<div class="flex justify-between border-t border-gray-200 pt-4 dark:border-gray-700 pr-6">
				<button
					onclick={resetToDefaults}
					class="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
				>
					<CarbonReset class="size-4" />
					Reset to Defaults
				</button>
				
				<div class="flex gap-3">
					<button
						onclick={handleClose}
						class="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
					>
						Cancel
					</button>
					<button
						onclick={saveConfiguration}
						disabled={saving}
						class="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
					>
						{#if saving}
							Saving...
						{:else}
							<CarbonCheckmark class="size-4" />
							Save Changes
						{/if}
					</button>
				</div>
			</div>
		{/if}
	</div>
</Modal>
