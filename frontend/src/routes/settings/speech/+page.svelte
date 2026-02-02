<script lang="ts">
	import { onMount } from "svelte";
	import CarbonTextToSpeech from "~icons/carbon/ibm-watson-text-to-speech";
	import CarbonMicrophone from "~icons/carbon/microphone";
	import CarbonSave from "~icons/carbon/save";
	import CarbonRenew from "~icons/carbon/renew";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonWarning from "~icons/carbon/warning";
	import CarbonPlay from "~icons/carbon/play";
	import CarbonDocument from "~icons/carbon/document";

	const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

	// Whisper STT configuration
	let whisperConfig = $state({
		model_size: 'base',
		device: 'auto',
		compute_type: 'int8',
		language: null as string | null,
		model_cache_dir: ''
	});

	// TTS configuration
	let ttsConfig = $state({
		default_voice: 'en_US-amy-low',
		audio_format: 'wav',
		default_speed: 1.0,
		use_gpu: false
	});

	let availableVoices = $state<any[]>([]);
	let savingTTS = $state(false);
	let loadingVoices = $state(false);
	let previewingVoice = $state(false);
	let currentPreviewAudio: HTMLAudioElement | null = null;

	// UI state
	let loading = $state(false);
	let saving = $state(false);
	let testing = $state(false);
	let error = $state<string | null>(null);
	let success = $state<string | null>(null);
	let activeSection = $state('stt');
	let requiresReload = $state(false);
	let healthStatus = $state<any>(null);

	// Test audio state
	let testAudioFile = $state<File | null>(null);
	let testTranscription = $state<string>('');

	const sections = [
		{ id: 'stt', label: 'Speech-to-Text (STT)' },
		{ id: 'tts', label: 'Text-to-Speech (TTS)' },
		{ id: 'test', label: 'Test & Diagnostics' }
	];

	const modelSizes = [
		{ value: 'tiny', label: 'Tiny (fastest, least accurate)', size: '~75MB' },
		{ value: 'base', label: 'Base (balanced)', size: '~150MB' },
		{ value: 'small', label: 'Small (good quality)', size: '~500MB' },
		{ value: 'medium', label: 'Medium (high quality)', size: '~1.5GB' },
		{ value: 'large', label: 'Large (best quality)', size: '~3GB' },
		{ value: 'large-v2', label: 'Large v2 (improved)', size: '~3GB' },
		{ value: 'large-v3', label: 'Large v3 (latest)', size: '~3GB' }
	];

	const devices = [
		{ value: 'auto', label: 'Auto-detect' },
		{ value: 'cpu', label: 'CPU only' },
		{ value: 'cuda', label: 'CUDA (GPU)' }
	];

	const computeTypes = [
		{ value: 'int8', label: 'INT8 (fastest, less accurate)' },
		{ value: 'float16', label: 'Float16 (balanced, requires GPU)' },
		{ value: 'float32', label: 'Float32 (most accurate, slowest)' }
	];

	const commonLanguages = [
		{ value: null, label: 'Auto-detect' },
		{ value: 'en', label: 'English' },
		{ value: 'es', label: 'Spanish' },
		{ value: 'fr', label: 'French' },
		{ value: 'de', label: 'German' },
		{ value: 'it', label: 'Italian' },
		{ value: 'pt', label: 'Portuguese' },
		{ value: 'nl', label: 'Dutch' },
		{ value: 'ja', label: 'Japanese' },
		{ value: 'ko', label: 'Korean' },
		{ value: 'zh', label: 'Chinese' },
		{ value: 'ru', label: 'Russian' },
		{ value: 'ar', label: 'Arabic' }
	];

	async function loadWhisperConfig() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/whisper`);
			
			if (!response.ok) {
				throw new Error('Failed to load Whisper configuration');
			}

			const data = await response.json();
			whisperConfig = data.config;
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to load Whisper config:', err);
		} finally {
			loading = false;
		}
	}

	async function loadTTSConfig() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/tts`);
			
			if (!response.ok) {
				throw new Error('Failed to load TTS configuration');
			}

			const data = await response.json();
			ttsConfig = {
				default_voice: data.config.default_voice,
				audio_format: data.config.audio_format,
				default_speed: data.config.default_speed,
				use_gpu: data.config.use_gpu
			};
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to load TTS config:', err);
		} finally {
			loading = false;
		}
	}

	async function loadAvailableVoices() {
		try {
			loadingVoices = true;
			console.log('Loading voices from:', `${BACKEND_URL}/api/speech/voices`);

			const response = await fetch(`${BACKEND_URL}/api/speech/voices`);
			
				if (!response.ok) {
					let serverMsg = '';
					try {
						const errJson = await response.json();
						serverMsg = errJson.detail || JSON.stringify(errJson);
					} catch (e) {
						serverMsg = await response.text();
					}
					console.error('Voice loading failed:', response.status, serverMsg);
					error = `Failed to load voices: ${serverMsg}`;
					return;
				}

				const data = await response.json();
				console.log('Voices loaded:', data);

				if (!data || !Array.isArray(data.voices)) {
					console.error('Unexpected voices payload', data);
					error = 'Unexpected voices payload from server';
					availableVoices = [];
					return;
				}

				// Normalize voice entries to a consistent shape so the UI can rely on fields
				availableVoices = data.voices.map((v: any) => ({
					voice_id: v.voice_id ?? v.id ?? v.name,
					name: v.name ?? v.voice_id ?? v.id ?? 'Unknown',
					language: v.language ?? v.lang ?? 'unknown',
					quality: v.quality ?? v.tier ?? 'medium',
					is_downloaded: v.is_downloaded ?? v.downloaded ?? v.local ?? false,
					model_size: v.model_size ?? v.size ?? '',
					gender: v.gender ?? '',
					description: v.description ?? v.note ?? ''
				}));
				console.log('Available voices count:', availableVoices.length);

			// If the configured default voice isn't present, pick the first available voice
			if (availableVoices.length > 0) {
				const found = availableVoices.find(v => v.voice_id === ttsConfig.default_voice);
				if (!found) {
					console.log('Default voice not found, using:', availableVoices[0].voice_id);
					ttsConfig = { ...ttsConfig, default_voice: availableVoices[0].voice_id };
				}
			}
		} catch (err) {
			console.error('Failed to load voices:', err);
			error = 'Failed to load TTS voices. Please check if backend is running.';
		} finally {
			loadingVoices = false;
		}
	}

	async function saveTTSConfig() {
		try {
			savingTTS = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/tts`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify(ttsConfig),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to save TTS configuration');
			}

			const data = await response.json();
			success = data.message;

			// Reload config to get updated values
			await loadTTSConfig();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to save TTS config:', err);
		} finally {
			savingTTS = false;
		}
	}

	async function changeVoice() {
		try {
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/voice`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					voice_id: ttsConfig.default_voice
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to change voice');
			}

			const data = await response.json();
			success = data.message;
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to change voice:', err);
		}
	}

	async function previewVoice() {
		try {
			previewingVoice = true;
			error = null;

			// Stop current preview if playing
			if (currentPreviewAudio) {
				currentPreviewAudio.pause();
				currentPreviewAudio = null;
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/preview-voice`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					voice_id: ttsConfig.default_voice,
					text: 'Hello, this is a voice preview. How does this sound?'
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to preview voice');
			}

			const audioBlob = await response.blob();
			const audioUrl = URL.createObjectURL(audioBlob);
			
			currentPreviewAudio = new Audio(audioUrl);
			currentPreviewAudio.play();

			// Clean up URL when audio finishes
			currentPreviewAudio.onended = () => {
				URL.revokeObjectURL(audioUrl);
			};

		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to preview voice:', err);
		} finally {
			previewingVoice = false;
		}
	}

	async function saveWhisperConfig() {
		try {
			saving = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/whisper`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					model_size: whisperConfig.model_size,
					device: whisperConfig.device,
					compute_type: whisperConfig.compute_type,
					language: whisperConfig.language || null,
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to save configuration');
			}

			const data = await response.json();
			requiresReload = data.requires_reload;
			success = data.message;

			// Reload config to get updated values
			await loadWhisperConfig();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to save Whisper config:', err);
		} finally {
			saving = false;
		}
	}

	async function reloadWhisperModel() {
		// Model reloads automatically on next transcription
		// Just clear the reload flag
		requiresReload = false;
		success = 'Configuration saved. Model will reload automatically on next use.';
	}

	async function checkHealth() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/health`);
			
			if (!response.ok) {
				throw new Error('Failed to check speech health');
			}

			healthStatus = await response.json();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to check speech health:', err);
		} finally {
			loading = false;
		}
	}

	async function testTranscribe() {
		if (!testAudioFile) {
			error = 'Please select an audio file to test';
			return;
		}

		try {
			testing = true;
			error = null;
			testTranscription = '';

			const formData = new FormData();
			formData.append('audio', testAudioFile);
			if (whisperConfig.language) {
				formData.append('language', whisperConfig.language);
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/transcribe`, {
				method: 'POST',
				body: formData,
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Transcription failed');
			}

			const data = await response.json();
			testTranscription = data.text;
			success = `Transcribed ${data.duration.toFixed(1)}s of ${data.language} audio`;
		} catch (err) {
			error = (err as Error).message;
			console.error('Transcription test failed:', err);
		} finally {
			testing = false;
		}
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			testAudioFile = target.files[0];
			testTranscription = '';
			error = null;
		}
	}

	onMount(() => {
		loadWhisperConfig();
		loadTTSConfig();
		loadAvailableVoices();
		checkHealth();
	});
</script>

<svelte:head>
	<title>Speech & Audio Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<!-- Header -->
	<div>
		<h1 class="text-2xl font-bold">Speech & Audio Settings</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Configure speech-to-text (STT) and text-to-speech (TTS) capabilities
		</p>
	</div>

	<!-- Tabs -->
	<div class="border-b border-gray-200 dark:border-gray-700">
		<nav class="-mb-px flex space-x-8">
			{#each sections as section}
				<button
					onclick={() => { activeSection = section.id; error = null; success = null; }}
					class="whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors
						{activeSection === section.id
							? 'border-blue-500 text-blue-600 dark:border-blue-400 dark:text-blue-400'
							: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:border-gray-600 dark:hover:text-gray-300'}"
				>
					{section.label}
				</button>
			{/each}
		</nav>
	</div>

	<!-- Alerts -->
	{#if error}
		<div class="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
			<div class="flex items-start gap-3">
				<CarbonWarning class="size-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<p class="text-sm font-medium text-red-800 dark:text-red-200">Error</p>
					<p class="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
				</div>
			</div>
		</div>
	{/if}

	{#if success}
		<div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
			<div class="flex items-start gap-3">
				<CarbonCheckmark class="size-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<p class="text-sm font-medium text-green-800 dark:text-green-200">Success</p>
					<p class="text-sm text-green-700 dark:text-green-300 mt-1">{success}</p>
				</div>
			</div>
		</div>
	{/if}

	{#if requiresReload}
		<div class="rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
			<div class="flex items-start justify-between gap-3">
				<div class="flex items-start gap-3">
					<CarbonWarning class="size-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
					<div class="flex-1">
						<p class="text-sm font-medium text-yellow-800 dark:text-yellow-200">Model Reload Required</p>
						<p class="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
							Configuration changes require reloading the Whisper model to take effect.
						</p>
					</div>
				</div>
				<button
					onclick={reloadWhisperModel}
					disabled={loading}
					class="rounded-lg bg-yellow-600 px-4 py-2 text-sm font-medium text-white hover:bg-yellow-700 disabled:opacity-50 dark:bg-yellow-500 dark:hover:bg-yellow-600"
				>
					<div class="flex items-center gap-2">
						<CarbonRenew class="size-4" />
						Reload Model
					</div>
				</button>
			</div>
		</div>
	{/if}

	<!-- Content Sections -->
	{#if activeSection === 'stt'}
		<!-- Speech-to-Text Configuration -->
		<div class="space-y-6">
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-blue-100 p-3 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
						<CarbonMicrophone class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Whisper STT Configuration</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Configure the Whisper speech-to-text model
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Model Size -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Model Size
						</label>
						<select
							bind:value={whisperConfig.model_size}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each modelSizes as size}
								<option value={size.value}>
									{size.label} - {size.size}
								</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Larger models provide better accuracy but require more memory and are slower.
						</p>
					</div>

					<!-- Device -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Compute Device
						</label>
						<select
							bind:value={whisperConfig.device}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each devices as device}
								<option value={device.value}>{device.label}</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Auto-detect will use GPU if available, otherwise CPU.
						</p>
					</div>

					<!-- Compute Type -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Compute Precision
						</label>
						<select
							bind:value={whisperConfig.compute_type}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each computeTypes as type}
								<option value={type.value}>{type.label}</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Float16 requires GPU. INT8 is recommended for CPU.
						</p>
					</div>

					<!-- Language -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Default Language
						</label>
						<select
							bind:value={whisperConfig.language}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each commonLanguages as lang}
								<option value={lang.value}>
									{lang.label}
								</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Set to auto-detect for multi-language support, or specify a language for better accuracy.
						</p>
					</div>

					<!-- Model Cache Directory -->
					{#if whisperConfig.model_cache_dir}
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
								Model Cache Directory
							</label>
							<div class="rounded-lg border border-gray-300 bg-gray-50 px-4 py-2.5 font-mono text-sm text-gray-700 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-300">
								{whisperConfig.model_cache_dir}
							</div>
							<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
								Downloaded Whisper models are cached here.
							</p>
						</div>
					{/if}
				</div>

				<!-- Save Button -->
				<div class="mt-8 flex justify-end">
					<button
						onclick={saveWhisperConfig}
						disabled={saving || loading}
						class="rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
					>
						<div class="flex items-center gap-2">
							{#if saving}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonSave class="size-4" />
							{/if}
							{saving ? 'Saving...' : 'Save Configuration'}
						</div>
					</button>
				</div>
			</div>

			<!-- Test Transcription -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="mb-6">
					<h2 class="text-lg font-semibold">Test Transcription</h2>
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Upload an audio file to test the Whisper STT system
					</p>
				</div>

				<div class="space-y-4">
					<!-- File Upload -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio File
						</label>
						<input
							type="file"
							accept="audio/*,.webm,.wav,.mp3,.m4a,.ogg,.flac"
							onchange={handleFileSelect}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 file:mr-4 file:rounded file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 dark:file:bg-blue-900/30 dark:file:text-blue-400"
						/>
						{#if testAudioFile}
							<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
								Selected: {testAudioFile.name} ({(testAudioFile.size / 1024).toFixed(1)} KB)
							</p>
						{/if}
					</div>

					<!-- Test Button -->
					<button
						onclick={testTranscribe}
						disabled={!testAudioFile || testing}
						class="w-full rounded-lg bg-green-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50 dark:bg-green-500 dark:hover:bg-green-600"
					>
						<div class="flex items-center justify-center gap-2">
							{#if testing}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonPlay class="size-4" />
							{/if}
							{testing ? 'Transcribing...' : 'Test Transcription'}
						</div>
					</button>

					<!-- Transcription Result -->
					{#if testTranscription}
						<div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
							<p class="text-sm font-medium text-green-800 dark:text-green-200 mb-2">
								Transcription Result:
							</p>
							<p class="text-sm text-green-700 dark:text-green-300 whitespace-pre-wrap">
								{testTranscription}
							</p>
						</div>
					{/if}
				</div>
			</div>
		</div>


		<!-- Info Box -->
		<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
			<div class="flex items-start gap-3">
				<CarbonDocument class="size-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<h3 class="font-semibold text-blue-900 dark:text-blue-200 text-sm">
						About Whisper STT
					</h3>
					<p class="text-sm text-blue-800 dark:text-blue-300 mt-1">
						Whisper is OpenAI's automatic speech recognition system. The first time you use a model size,
						it will be downloaded automatically. Larger models provide better accuracy but require more
						resources. The 'base' model is a good balance for most users.
					</p>
				</div>
			</div>
		</div>

	{:else if activeSection === 'tts'}
		<!-- Text-to-Speech Configuration -->
		<div class="space-y-6">
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-purple-100 p-3 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
						<CarbonTextToSpeech class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Text-to-Speech Configuration</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Configure Piper TTS voices and audio settings
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Voice Selection -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Default Voice
						</label>
						<div class="flex gap-3">
							<select
								bind:value={ttsConfig.default_voice}
								onchange={changeVoice}
								disabled={loadingVoices}
								class="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							>
								{#if loadingVoices}
									<option>Loading voices...</option>
								{:else if availableVoices.length === 0}
									<option>No voices available</option>
								{:else}
									{#each availableVoices as voice}
										<option value={voice.voice_id}>
											{voice.name} ({voice.language}) - {voice.quality}
											{#if voice.is_downloaded}✓{/if}
										</option>
									{/each}
								{/if}
							</select>
							<button
								onclick={previewVoice}
								disabled={previewingVoice || loadingVoices || availableVoices.length === 0}
								class="rounded-lg bg-purple-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600 whitespace-nowrap"
							>
								<div class="flex items-center gap-2">
									{#if previewingVoice}
										<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
									{:else}
										<CarbonPlay class="size-4" />
									{/if}
									Preview
								</div>
							</button>
						</div>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Select a voice and click Preview to hear a sample. Voices marked with ✓ are downloaded locally.
						</p>
					</div>

					<!-- Voice Info -->
					{#if !loadingVoices && availableVoices.length > 0}
						{@const selectedVoice = availableVoices.find(v => v.voice_id === ttsConfig.default_voice)}
						{#if selectedVoice}
							<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50">
								<div class="grid grid-cols-2 gap-4 text-sm">
									<div>
										<span class="text-gray-600 dark:text-gray-400">Gender:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100 capitalize">{selectedVoice.gender}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Quality:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100 capitalize">{selectedVoice.quality}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Language:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100">{selectedVoice.language}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Model Size:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100">{selectedVoice.model_size}</span>
									</div>
								</div>
								<p class="text-xs text-gray-600 dark:text-gray-400 mt-3">
									{selectedVoice.description}
								</p>
							</div>
						{/if}
					{/if}

					<!-- Speed Control -->
					<div>
						<div class="flex justify-between items-center mb-2">
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
								Speech Speed
							</label>
							<span class="text-sm font-mono text-gray-600 dark:text-gray-400">
								{ttsConfig.default_speed.toFixed(1)}x
							</span>
						</div>
						<input 
							type="range" 
							min="0.0" 
							max="2.0" 
							step="0.1"
							bind:value={ttsConfig.default_speed}
							class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-purple-600"
						/>
						<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
							<span>0.0x (Slowest)</span>
							<span>1.0x (Normal)</span>
							<span>2.0x (Fastest)</span>
						</div>
					</div>

					<!-- Audio Format -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio Format
						</label>
						<select
							bind:value={ttsConfig.audio_format}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="wav">WAV (Uncompressed, Higher Quality)</option>
							<option value="mp3">MP3 (Compressed, Smaller Size)</option>
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							WAV provides best quality but larger file sizes. MP3 is compressed and more efficient.
						</p>
					</div>

					<!-- GPU Acceleration -->
					<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50">
						<label class="flex items-start gap-3 cursor-pointer">
							<input 
								type="checkbox" 
								bind:checked={ttsConfig.use_gpu}
								class="mt-0.5 size-4 rounded border-gray-300 text-purple-600 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700"
							/>
							<div class="flex-1">
								<div class="font-medium text-gray-900 dark:text-gray-100">
									Enable GPU Acceleration (CUDA)
								</div>
								<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
									Use GPU for faster TTS synthesis. Requires NVIDIA GPU with CUDA support.
									Disable if you encounter errors or don't have a compatible GPU.
								</p>
							</div>
						</label>
					</div>
				</div>

				<!-- Save Button -->
				<div class="mt-8 flex justify-end">
					<button
						onclick={saveTTSConfig}
						disabled={savingTTS || loading}
						class="rounded-lg bg-purple-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600"
					>
						<div class="flex items-center gap-2">
							{#if savingTTS}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonSave class="size-4" />
							{/if}
							{savingTTS ? 'Saving...' : 'Save Configuration'}
						</div>
					</button>
				</div>
			</div>

			<!-- Info Box -->
			<div class="rounded-lg border border-purple-200 bg-purple-50 p-4 dark:border-purple-800 dark:bg-purple-900/20">
				<div class="flex items-start gap-3">
					<CarbonDocument class="size-5 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
					<div class="flex-1">
						<h3 class="font-semibold text-purple-900 dark:text-purple-200 text-sm">
							About Piper TTS
						</h3>
						<p class="text-sm text-purple-800 dark:text-purple-300 mt-1">
							Piper is a fast, local neural text-to-speech system. Voice models are automatically loaded on first use.
							The system supports multiple languages and voice qualities. For best performance on CPU, use low or medium quality voices.
						</p>
					</div>
				</div>
			</div>
		</div>

	{:else if activeSection === 'test'}
		<!-- Test & Diagnostics -->
		<div class="space-y-6">
			<!-- Health Status -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center justify-between mb-6">
					<div>
						<h2 class="text-lg font-semibold">Speech System Health</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Check the status of speech services
						</p>
					</div>
					<button
						onclick={checkHealth}
						disabled={loading}
						class="rounded-lg bg-gray-600 px-4 py-2 text-sm font-medium text-white hover:bg-gray-700 disabled:opacity-50 dark:bg-gray-500 dark:hover:bg-gray-600"
					>
						<div class="flex items-center gap-2">
							<CarbonRenew class="size-4" />
							Refresh
						</div>
					</button>
				</div>

				{#if loading && !healthStatus}
					<div class="flex items-center justify-center py-12">
						<div class="size-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent"></div>
					</div>
				{:else if healthStatus}
					<div class="space-y-4">
						<div class="flex items-center justify-between rounded-lg border p-4 dark:border-gray-700">
							<div class="flex items-center gap-3">
								{#if healthStatus.whisper_available}
									<div class="size-3 rounded-full bg-green-500"></div>
								{:else}
									<div class="size-3 rounded-full bg-red-500"></div>
								{/if}
								<div>
									<p class="font-medium">Whisper STT</p>
									<p class="text-sm text-gray-600 dark:text-gray-400">
										{healthStatus.whisper_available ? 'Available' : 'Not available'}
									</p>
								</div>
							</div>
							{#if healthStatus.whisper_config}
								<div class="text-right text-sm text-gray-600 dark:text-gray-400">
									<p>Model: {healthStatus.whisper_config.model_size}</p>
									<p>Device: {healthStatus.whisper_config.device}</p>
								</div>
							{/if}
						</div>

						<div class="flex items-center justify-between rounded-lg border p-4 dark:border-gray-700">
							<div class="flex items-center gap-3">
								{#if healthStatus && healthStatus.tts_available !== undefined}
									{#if healthStatus.tts_available}
										<div class="size-3 rounded-full bg-green-500"></div>
									{:else}
										<div class="size-3 rounded-full bg-red-500"></div>
									{/if}
									<div>
										<p class="font-medium">{healthStatus.tts_engine || 'Piper-TTS'}</p>
										<p class="text-sm text-gray-600 dark:text-gray-400">
											{healthStatus.tts_available ? 'Active' : 'Not available'}
										</p>
									</div>
								{:else}
									<div class="size-3 rounded-full bg-green-500"></div>
									<div>
										<p class="font-medium">Piper-TTS</p>
										<p class="text-sm text-gray-600 dark:text-gray-400">Active</p>
									</div>
								{/if}
							</div>						<button
							class="rounded-lg bg-purple-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600"
							disabled
							title="Engine switching coming soon (Qwen3-TTS)"
						>
							Change Engine
						</button>						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>
