<script lang="ts">
	import { onMount } from 'svelte';
	import Navigation from '$lib/components/Navigation.svelte';
	import { systemConfig, gpuInfo, systemStatus } from '$lib/stores/system';
	import { systemApi } from '$lib/utils/api';
	import { Save, Cpu, HardDrive, Zap, Settings as SettingsIcon } from 'lucide-svelte';
	
	let isLoading = false;
	let saveMessage = '';
	let localConfig: any = {};
	
	onMount(async () => {
		// Load current configuration
		try {
			const config = await systemApi.getConfig();
			localConfig = { ...config };
			systemConfig.set(config);
		} catch (error) {
			console.error('Failed to load configuration:', error);
		}
	});
	
	const saveConfiguration = async () => {
		isLoading = true;
		saveMessage = '';
		
		try {
			const updatedConfig = await systemApi.updateConfig(localConfig);
			systemConfig.set(updatedConfig);
			saveMessage = 'Configuration saved successfully!';
			
			setTimeout(() => {
				saveMessage = '';
			}, 3000);
			
		} catch (error) {
			console.error('Save failed:', error);
			saveMessage = 'Failed to save configuration.';
		} finally {
			isLoading = false;
		}
	};
</script>

<Navigation />

<div class="container mx-auto max-w-4xl p-6">
	<div class="mb-8">
		<h1 class="text-3xl font-bold mb-2 flex items-center gap-3">
			<SettingsIcon size={32} />
			System Settings
		</h1>
		<p class="text-base-content/70">Configure your RAG assistant's performance and behavior.</p>
	</div>
	
	{#if saveMessage}
		<div class="alert {saveMessage.includes('success') ? 'alert-success' : 'alert-error'} mb-6">
			{saveMessage}
		</div>
	{/if}
	
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<!-- Performance Settings -->
		<div class="card bg-base-100 shadow-lg">
			<div class="card-body">
				<h2 class="card-title flex items-center gap-2">
					<Cpu size={20} />
					Performance
				</h2>
				
				<div class="form-control">
					<label class="label" for="cpu-usage">
						<span class="label-text">CPU Usage Limit (%)</span>
					</label>
					<input 
						id="cpu-usage"
						type="range" 
						min="25" 
						max="100" 
						step="5"
						bind:value={localConfig.max_cpu_usage_percent}
						class="range range-primary" />
					<div class="w-full flex justify-between text-xs px-2">
						<span>25%</span>
						<span>50%</span>
						<span>75%</span>
						<span>100%</span>
					</div>
					<div class="text-center font-mono text-lg mt-2">
						{localConfig.max_cpu_usage_percent}%
					</div>
				</div>
				
				<div class="form-control">
					<label class="label" for="memory-limit">
						<span class="label-text">Memory Limit (GB)</span>
					</label>
					<input 
						id="memory-limit"
						type="number" 
						min="1" 
						max="64" 
						bind:value={localConfig.max_memory_usage_gb}
						class="input input-bordered" />
				</div>
			</div>
		</div>
		
		<!-- GPU Settings -->
		<div class="card bg-base-100 shadow-lg">
			<div class="card-body">
				<h2 class="card-title flex items-center gap-2">
					<Zap size={20} />
					GPU Acceleration
				</h2>
				
				{#if $gpuInfo?.available}
					<div class="alert alert-success mb-4">
						<div>
							<h3 class="font-bold">GPU Available!</h3>
							<div class="text-sm">
								{$gpuInfo.device_count} GPU(s) detected
								{#if $gpuInfo.driver_version}
									<br>Driver: v{$gpuInfo.driver_version}
								{/if}
							</div>
						</div>
					</div>
					
					<div class="form-control">
						<label class="cursor-pointer label">
							<span class="label-text">Enable GPU Acceleration</span>
							<input 
								type="checkbox" 
								bind:checked={localConfig.enable_gpu_acceleration}
								class="checkbox checkbox-primary" />
						</label>
					</div>
					
					{#if localConfig.enable_gpu_acceleration}
						<div class="form-control">
							<label class="label">
								<span class="label-text">GPU Memory Limit (MB)</span>
							</label>
							<input 
								type="number" 
								min="512" 
								max={$gpuInfo.memory_total || 8192}
								bind:value={localConfig.gpu_memory_limit_mb}
								class="input input-bordered" />
							{#if $gpuInfo.memory_total}
								<label class="label">
									<span class="label-text-alt">
										Total: {$gpuInfo.memory_total}MB, 
										Used: {$gpuInfo.memory_used || 0}MB
									</span>
								</label>
							{/if}
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Preferred GPU Device</span>
							</label>
							<select 
								bind:value={localConfig.preferred_gpu_device}
								class="select select-bordered">
								{#each Array($gpuInfo.device_count || 1) as _, i}
									<option value={i}>GPU {i}</option>
								{/each}
							</select>
						</div>
					{/if}
				{:else}
					<div class="alert alert-warning">
						<div>
							<h3 class="font-bold">No GPU Detected</h3>
							<div class="text-sm">
								GPU acceleration is not available on this system.
								Running in CPU-only mode.
							</div>
						</div>
					</div>
				{/if}
			</div>
		</div>
		
		<!-- Document Processing -->
		<div class="card bg-base-100 shadow-lg lg:col-span-2">
			<div class="card-body">
				<h2 class="card-title flex items-center gap-2">
					<HardDrive size={20} />
					Document Processing
				</h2>
				
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
					<div class="space-y-4">
						<div class="form-control">
							<label class="cursor-pointer label">
								<span class="label-text">Auto-Indexing</span>
								<input 
									type="checkbox" 
									bind:checked={localConfig.auto_indexing}
									class="checkbox checkbox-primary" />
							</label>
							<label class="label">
								<span class="label-text-alt">Automatically process new documents</span>
							</label>
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Indexing Interval (minutes)</span>
							</label>
							<input 
								type="number" 
								min="5" 
								max="1440"
								bind:value={localConfig.indexing_interval_minutes}
								class="input input-bordered" />
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Chunk Size</span>
							</label>
							<input 
								type="number" 
								min="100" 
								max="2000"
								bind:value={localConfig.chunk_size}
								class="input input-bordered" />
							<label class="label">
								<span class="label-text-alt">Characters per chunk</span>
							</label>
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Chunk Overlap</span>
							</label>
							<input 
								type="number" 
								min="0" 
								max="500"
								bind:value={localConfig.chunk_overlap}
								class="input input-bordered" />
							<label class="label">
								<span class="label-text-alt">Character overlap between chunks</span>
							</label>
						</div>
					</div>
					
					<div class="space-y-4">
						<div class="form-control">
							<label class="label">
								<span class="label-text">Watch Directories</span>
							</label>
							<textarea 
								bind:value={localConfig.watch_directories}
								placeholder="One directory path per line"
								class="textarea textarea-bordered h-20"></textarea>
							<label class="label">
								<span class="label-text-alt">Directories to monitor for new files</span>
							</label>
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Excluded Extensions</span>
							</label>
							<textarea 
								bind:value={localConfig.excluded_extensions}
								placeholder=".tmp, .log, .cache"
								class="textarea textarea-bordered h-20"></textarea>
							<label class="label">
								<span class="label-text-alt">File extensions to ignore</span>
							</label>
						</div>
						
						<div class="form-control">
							<label class="label">
								<span class="label-text">Vector Dimensions</span>
							</label>
							<select 
								bind:value={localConfig.vector_dimensions}
								class="select select-bordered">
								<option value={384}>384 (Fast)</option>
								<option value={768}>768 (Balanced)</option>
								<option value={1024}>1024 (Accurate)</option>
							</select>
							<label class="label">
								<span class="label-text-alt">Higher dimensions = better accuracy, slower processing</span>
							</label>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<!-- System Status -->
		<div class="card bg-base-100 shadow-lg lg:col-span-2">
			<div class="card-body">
				<h2 class="card-title">System Status</h2>
				
				<div class="stats stats-vertical lg:stats-horizontal">
					<div class="stat">
						<div class="stat-title">Configuration</div>
						<div class="stat-value text-lg {$systemStatus.configured ? 'text-success' : 'text-error'}">
							{$systemStatus.configured ? 'Loaded' : 'Error'}
						</div>
					</div>
					
					<div class="stat">
						<div class="stat-title">GPU Status</div>
						<div class="stat-value text-lg {$systemStatus.gpuAvailable ? 'text-success' : 'text-warning'}">
							{$systemStatus.gpuAvailable ? 'Available' : 'CPU Only'}
						</div>
					</div>
					
					<div class="stat">
						<div class="stat-title">System Health</div>
						<div class="stat-value text-lg {$systemStatus.healthy ? 'text-success' : 'text-error'}">
							{$systemStatus.healthy ? 'Healthy' : 'Issues'}
						</div>
					</div>
				</div>
				
				{#if $systemStatus.services}
					<div class="mt-4">
						<h3 class="font-semibold mb-2">Services Status:</h3>
						<div class="flex flex-wrap gap-2">
							{#each Object.entries($systemStatus.services) as [service, status]}
								<div class="badge {status ? 'badge-success' : 'badge-error'}">
									{service}: {status ? 'Running' : 'Down'}
								</div>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Save Button -->
	<div class="flex justify-center mt-8">
		<button 
			on:click={saveConfiguration}
			disabled={isLoading}
			class="btn btn-primary btn-lg">
			{#if isLoading}
				<span class="loading loading-spinner"></span>
			{:else}
				<Save size={20} />
			{/if}
			Save Configuration
		</button>
	</div>
</div>
