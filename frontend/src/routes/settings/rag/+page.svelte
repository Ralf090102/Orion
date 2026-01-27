<script lang="ts">
	import CarbonSave from "~icons/carbon/save";
	import CarbonReset from "~icons/carbon/reset";
	import CarbonWarning from "~icons/carbon/warning";

	// Placeholder data - will be loaded from backend later
	let settings = $state({
		embedding: {
			model: "all-MiniLM-L12-v2",
			batch_size: 64,
			timeout: 30,
			cache_embeddings: true
		},
		chunking: {
			strategy: "recursive",
			chunk_size: 512,
			chunk_overlap: 128,
			max_chunk_size: 512,
			min_chunk_size: 256
		},
		retrieval: {
			default_k: 5,
			max_k: 20,
			similarity_threshold: 0.2,
			enable_reranking: true,
			enable_hybrid_search: true,
			semantic_weight: 0.8,
			keyword_weight: 0.2,
			enable_mmr: true,
			mmr_diversity_bias: 0.5,
			mmr_fetch_k: 20,
			mmr_threshold: 0.475
		},
		reranker: {
			model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
			batch_size: 16,
			timeout: 30,
			top_k: 10,
			score_threshold: 0.5
		},
		generation: {
			mode: "rag",
			enable_citations: true,
			citation_format: "[{index}]",
			max_context_chunks: 5,
			validate_citations: true,
			expand_citations: false,
			max_history_messages: 10,
			enable_rag_augmentation: true,
			rag_trigger_mode: "auto",
			max_total_tokens: 4096,
			reserve_tokens_for_response: 1024
		},
		vectorstore: {
			collection_name: "orion_knowledge_base",
			persist_directory: "./data/chroma-data",
			distance_metric: "cosine",
			batch_size: 64
		},
		gpu: {
			enabled: false,
			auto_detect: true,
			preferred_device: "auto",
			fallback_to_cpu: true
		}
	});

	let saved = $state(false);
	let loading = $state(false);
	let activeSection = $state('embedding');

	const sections = [
		{ id: 'embedding', label: 'Embedding' },
		{ id: 'chunking', label: 'Chunking' },
		{ id: 'retrieval', label: 'Retrieval' },
		{ id: 'reranker', label: 'Reranker' },
		{ id: 'generation', label: 'Generation' },
		{ id: 'vectorstore', label: 'Vector Store' },
		{ id: 'gpu', label: 'GPU' }
	];

	function scrollToSection(sectionId: string) {
		activeSection = sectionId;
	}

	async function saveSettings() {
		loading = true;
		try {
			// TODO: POST to /api/settings
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
		// Reset to default values
		settings = {
			embedding: {
				model: "all-MiniLM-L12-v2",
				batch_size: 64,
				timeout: 30,
				cache_embeddings: true
			},
			chunking: {
				strategy: "recursive",
				chunk_size: 512,
				chunk_overlap: 128,
				max_chunk_size: 512,
				min_chunk_size: 256
			},
			retrieval: {
				default_k: 5,
				max_k: 20,
				similarity_threshold: 0.2,
				enable_reranking: true,
				enable_hybrid_search: true,
				semantic_weight: 0.8,
				keyword_weight: 0.2,
				enable_mmr: true,
				mmr_diversity_bias: 0.5,
				mmr_fetch_k: 20,
				mmr_threshold: 0.475
			},
			reranker: {
				model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
				batch_size: 16,
				timeout: 30,
				top_k: 10,
				score_threshold: 0.5
			},
			generation: {
				mode: "rag",
				enable_citations: true,
				citation_format: "[{index}]",
				max_context_chunks: 5,
				validate_citations: true,
				expand_citations: false,
				max_history_messages: 10,
				enable_rag_augmentation: true,
				rag_trigger_mode: "auto",
				max_total_tokens: 4096,
				reserve_tokens_for_response: 1024
			},
			vectorstore: {
				collection_name: "orion_knowledge_base",
				persist_directory: "./data/chroma-data",
				distance_metric: "cosine",
				batch_size: 64
			},
			gpu: {
				enabled: false,
				auto_detect: true,
				preferred_device: "auto",
				fallback_to_cpu: true
			}
		};
	}
</script>

<svelte:head>
	<title>RAG Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<div>
		<h1 class="text-2xl font-bold">RAG Pipeline Settings</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Configure the Retrieval-Augmented Generation system
		</p>
	</div>

	<!-- Warning Banner -->
	<div class="rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-900/20">
		<div class="flex items-start gap-3">
			<CarbonWarning class="size-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
			<div class="flex-1">
				<h3 class="font-semibold text-amber-900 dark:text-amber-200 text-sm">
					⚠️ Advanced Configuration
				</h3>
				<p class="text-sm text-amber-800 dark:text-amber-300 mt-1">
					Do not modify these settings unless you understand their impact. Incorrect configuration may break the RAG pipeline and affect retrieval quality.
				</p>
			</div>
		</div>
	</div>

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

	<form onsubmit={(e) => { e.preventDefault(); saveSettings(); }} class="flex flex-col gap-6">
		<!-- Embedding Settings -->
		{#if activeSection === 'embedding'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Embedding</h2>
			
			<div class="flex flex-col gap-4">
				<div>
					<label for="embedding-model" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Model
					</label>
					<input
						type="text"
						id="embedding-model"
						bind:value={settings.embedding.model}
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Embedding model for vector generation</p>
				</div>

				<div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
					<div>
						<label for="embedding-batch-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Batch Size
						</label>
						<input
							type="number"
							id="embedding-batch-size"
							bind:value={settings.embedding.batch_size}
							min="1"
							max="256"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="embedding-timeout" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Timeout (seconds)
						</label>
						<input
							type="number"
							id="embedding-timeout"
							bind:value={settings.embedding.timeout}
							min="10"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div class="flex items-center">
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={settings.embedding.cache_embeddings}
								class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
							/>
							<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Cache Embeddings</span>
						</label>
					</div>
				</div>
			</div>
		</section>
		{/if}

		<!-- Chunking Settings -->
		{#if activeSection === 'chunking'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Chunking</h2>
			
			<div class="flex flex-col gap-4">
				<div>
					<label for="chunking-strategy" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Strategy
					</label>
					<select
						id="chunking-strategy"
						bind:value={settings.chunking.strategy}
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					>
						<option value="recursive">Recursive</option>
						<option value="semantic">Semantic</option>
						<option value="smart">Smart</option>
					</select>
					<p class="mt-1 text-xs text-gray-500">Method for splitting documents into chunks</p>
				</div>

				<div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
					<div>
						<label for="chunk-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Chunk Size
						</label>
						<input
							type="number"
							id="chunk-size"
							bind:value={settings.chunking.chunk_size}
							min="100"
							max="2048"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="chunk-overlap" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Chunk Overlap
						</label>
						<input
							type="number"
							id="chunk-overlap"
							bind:value={settings.chunking.chunk_overlap}
							min="0"
							max="512"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="max-chunk-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max Chunk Size
						</label>
						<input
							type="number"
							id="max-chunk-size"
							bind:value={settings.chunking.max_chunk_size}
							min="100"
							max="4096"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="min-chunk-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Min Chunk Size
						</label>
						<input
							type="number"
							id="min-chunk-size"
							bind:value={settings.chunking.min_chunk_size}
							min="50"
							max="1024"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>
			</div>
		</section>
		{/if}

		<!-- Retrieval Settings -->
		{#if activeSection === 'retrieval'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Retrieval</h2>
			
			<div class="flex flex-col gap-4">
				<div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
					<div>
						<label for="default-k" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Default K
						</label>
						<input
							type="number"
							id="default-k"
							bind:value={settings.retrieval.default_k}
							min="1"
							max="50"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="mt-1 text-xs text-gray-500">Documents to retrieve</p>
					</div>

					<div>
						<label for="max-k" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max K
						</label>
						<input
							type="number"
							id="max-k"
							bind:value={settings.retrieval.max_k}
							min="1"
							max="100"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="similarity-threshold" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Similarity Threshold
						</label>
						<input
							type="number"
							id="similarity-threshold"
							bind:value={settings.retrieval.similarity_threshold}
							min="0"
							max="1"
							step="0.01"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>

				<div class="flex flex-wrap gap-4">
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.retrieval.enable_reranking}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable Reranking</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.retrieval.enable_hybrid_search}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable Hybrid Search</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.retrieval.enable_mmr}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable MMR</span>
					</label>
				</div>

				{#if settings.retrieval.enable_hybrid_search}
					<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
						<div>
							<label for="semantic-weight" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
								Semantic Weight
							</label>
							<input
								type="number"
								id="semantic-weight"
								bind:value={settings.retrieval.semantic_weight}
								min="0"
								max="1"
								step="0.1"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							/>
						</div>

						<div>
							<label for="keyword-weight" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
								Keyword Weight
							</label>
							<input
								type="number"
								id="keyword-weight"
								bind:value={settings.retrieval.keyword_weight}
								min="0"
								max="1"
								step="0.1"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							/>
						</div>
					</div>
				{/if}

				{#if settings.retrieval.enable_mmr}
					<div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
						<div>
							<label for="mmr-diversity-bias" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
								MMR Diversity Bias
							</label>
							<input
								type="number"
								id="mmr-diversity-bias"
								bind:value={settings.retrieval.mmr_diversity_bias}
								min="0"
								max="1"
								step="0.1"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							/>
						</div>

						<div>
							<label for="mmr-fetch-k" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
								MMR Fetch K
							</label>
							<input
								type="number"
								id="mmr-fetch-k"
								bind:value={settings.retrieval.mmr_fetch_k}
								min="1"
								max="200"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							/>
						</div>

						<div>
							<label for="mmr-threshold" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
								MMR Threshold
							</label>
							<input
								type="number"
								id="mmr-threshold"
								bind:value={settings.retrieval.mmr_threshold}
								min="0"
								max="1"
								step="0.01"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							/>
						</div>
					</div>
				{/if}
			</div>
		</section>
		{/if}

		<!-- Reranker Settings -->
		{#if activeSection === 'reranker'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Reranker</h2>
			
			<div class="flex flex-col gap-4">
				<div>
					<label for="reranker-model" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Model
					</label>
					<input
						type="text"
						id="reranker-model"
						bind:value={settings.reranker.model}
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Cross-encoder model for reranking</p>
				</div>

				<div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
					<div>
						<label for="reranker-batch-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Batch Size
						</label>
						<input
							type="number"
							id="reranker-batch-size"
							bind:value={settings.reranker.batch_size}
							min="1"
							max="128"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="reranker-timeout" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Timeout (seconds)
						</label>
						<input
							type="number"
							id="reranker-timeout"
							bind:value={settings.reranker.timeout}
							min="10"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="reranker-top-k" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Top K
						</label>
						<input
							type="number"
							id="reranker-top-k"
							bind:value={settings.reranker.top_k}
							min="1"
							max="50"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="reranker-score-threshold" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Score Threshold
						</label>
						<input
							type="number"
							id="reranker-score-threshold"
							bind:value={settings.reranker.score_threshold}
							min="0"
							max="1"
							step="0.1"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>
			</div>
		</section>
		{/if}

		<!-- Generation Settings -->
		{#if activeSection === 'generation'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Generation</h2>
			
			<div class="flex flex-col gap-4">
				<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
					<div>
						<label for="generation-mode" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Mode
						</label>
						<select
							id="generation-mode"
							bind:value={settings.generation.mode}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="rag">RAG</option>
							<option value="chat">Chat</option>
						</select>
					</div>

					<div>
						<label for="citation-format" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Citation Format
						</label>
						<input
							type="text"
							id="citation-format"
							bind:value={settings.generation.citation_format}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>

				<div class="flex flex-wrap gap-4">
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.generation.enable_citations}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable Citations</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.generation.validate_citations}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Validate Citations</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.generation.expand_citations}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Expand Citations</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.generation.enable_rag_augmentation}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable RAG Augmentation</span>
					</label>
				</div>

				<div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
					<div>
						<label for="max-context-chunks" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max Context Chunks
						</label>
						<input
							type="number"
							id="max-context-chunks"
							bind:value={settings.generation.max_context_chunks}
							min="1"
							max="20"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="max-history-messages" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max History Messages
						</label>
						<input
							type="number"
							id="max-history-messages"
							bind:value={settings.generation.max_history_messages}
							min="1"
							max="50"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="max-total-tokens" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Max Total Tokens
						</label>
						<input
							type="number"
							id="max-total-tokens"
							bind:value={settings.generation.max_total_tokens}
							min="512"
							max="32768"
							step="512"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="reserve-tokens" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Reserve Tokens
						</label>
						<input
							type="number"
							id="reserve-tokens"
							bind:value={settings.generation.reserve_tokens_for_response}
							min="128"
							max="4096"
							step="128"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>

				<div>
					<label for="rag-trigger-mode" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						RAG Trigger Mode
					</label>
					<select
						id="rag-trigger-mode"
						bind:value={settings.generation.rag_trigger_mode}
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					>
						<option value="always">Always</option>
						<option value="auto">Auto</option>
						<option value="manual">Manual</option>
						<option value="never">Never</option>
					</select>
				</div>
			</div>
		</section>
		{/if}

		<!-- Vector Store Settings -->
		{#if activeSection === 'vectorstore'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Vector Store</h2>
			
			<div class="flex flex-col gap-4">
				<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
					<div>
						<label for="collection-name" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Collection Name
						</label>
						<input
							type="text"
							id="collection-name"
							bind:value={settings.vectorstore.collection_name}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>

					<div>
						<label for="persist-directory" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Persist Directory
						</label>
						<input
							type="text"
							id="persist-directory"
							bind:value={settings.vectorstore.persist_directory}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>

				<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
					<div>
						<label for="distance-metric" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Distance Metric
						</label>
						<select
							id="distance-metric"
							bind:value={settings.vectorstore.distance_metric}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="cosine">Cosine</option>
							<option value="l2">L2 (Euclidean)</option>
							<option value="ip">Inner Product</option>
						</select>
					</div>

					<div>
						<label for="vectorstore-batch-size" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
							Batch Size
						</label>
						<input
							type="number"
							id="vectorstore-batch-size"
							bind:value={settings.vectorstore.batch_size}
							min="1"
							max="256"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
					</div>
				</div>
			</div>
		</section>
		{/if}

		<!-- GPU Settings -->
		{#if activeSection === 'gpu'}
		<section class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
			<h2 class="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">GPU Acceleration</h2>
			
			<div class="flex flex-col gap-4">
				<div class="flex flex-wrap gap-4">
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.gpu.enabled}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Enable GPU</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.gpu.auto_detect}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Auto Detect</span>
					</label>

					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={settings.gpu.fallback_to_cpu}
							class="size-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
						/>
						<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Fallback to CPU</span>
					</label>
				</div>

				<div>
					<label for="preferred-device" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
						Preferred Device
					</label>
					<input
						type="text"
						id="preferred-device"
						bind:value={settings.gpu.preferred_device}
						placeholder="auto, cpu, cuda:0, etc."
						class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
					/>
					<p class="mt-1 text-xs text-gray-500">Device identifier (auto, cpu, cuda:0, etc.)</p>
				</div>
			</div>
		</section>
		{/if}

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
			<div class="rounded-lg border border-green-200 bg-green-50 p-4 text-sm text-green-800 dark:border-green-800 dark:bg-green-900/20 dark:text-green-400 flex items-center gap-2">
				<CarbonSave class="size-4" />
				Settings saved successfully
			</div>
		{/if}
	</form>
</div>
