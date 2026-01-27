<script lang="ts">
	import CarbonSettings from "~icons/carbon/settings";
	import CarbonChevronRight from "~icons/carbon/chevron-right";
	import CarbonDocumentTasks from "~icons/carbon/document-tasks";

	const settingsCategories = [
		{
			id: 'rag',
			title: 'RAG Pipeline',
			description: 'Configure retrieval, embedding, chunking, and generation settings',
			icon: CarbonDocumentTasks,
			path: '/settings/rag',
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
