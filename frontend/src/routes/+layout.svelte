<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { systemApi } from '$lib/utils/api';
	import { systemConfig, gpuInfo, currentProfile } from '$lib/stores/system';

	let { children } = $props();

	// Initialize system data on app startup
	onMount(async () => {
		try {
			const [config, gpu, profiles] = await Promise.all([
				systemApi.getConfig(),
				systemApi.getGPUInfo(),
				systemApi.getProfiles()
			]);
			
			systemConfig.set(config);
			gpuInfo.set(gpu);
			
			if (profiles.length > 0) {
				currentProfile.set(profiles[0]);
			}
		} catch (error) {
			console.error('Failed to initialize system:', error);
		}
	});
</script>

<svelte:head>
	<title>Orion - Personal RAG Assistant</title>
	<meta name="description" content="Your intelligent document assistant" />
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
</svelte:head>

<div data-theme="orion" class="min-h-screen bg-base-100">
	{@render children?.()}
</div>
