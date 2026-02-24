<script lang="ts">
	import {
		convoModeState,
		updateSettings,
		resetSettings,
		type InputMode,
	} from "$lib/stores/conversationMode.svelte";
	
	import CarbonReset from "~icons/carbon/reset";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	
	interface Props {
		/** Compact mode for popover, full mode for settings page */
		compact?: boolean;
		/** Callback when settings are changed */
		onchange?: () => void;
	}
	
	let { compact = true, onchange }: Props = $props();
	
	// Local state bound to form controls
	let localSettings = $state({
		autoTTS: convoModeState.settings.autoTTS,
		inputMode: convoModeState.settings.inputMode as InputMode,
		silenceDuration: convoModeState.settings.silenceDuration,
		autoResume: convoModeState.settings.autoResume,
		disableRAG: convoModeState.settings.disableRAG,
		sttLanguage: convoModeState.settings.sttLanguage || 'en',
	});
	
	// Sync local state when store changes
	$effect(() => {
		localSettings = {
			autoTTS: convoModeState.settings.autoTTS,
			inputMode: convoModeState.settings.inputMode as InputMode,
			silenceDuration: convoModeState.settings.silenceDuration,
			autoResume: convoModeState.settings.autoResume,
			disableRAG: convoModeState.settings.disableRAG,
			sttLanguage: convoModeState.settings.sttLanguage || 'en',
		};
	});
	
	function handleToggle(key: 'autoTTS' | 'autoResume' | 'disableRAG') {
		const newValue = !localSettings[key];
		localSettings[key] = newValue;
		updateSettings({ [key]: newValue });
		onchange?.();
	}
	
	function handleInputModeChange(event: Event) {
		const target = event.target as HTMLSelectElement;
		const newMode = target.value as InputMode;
		localSettings.inputMode = newMode;
		updateSettings({ inputMode: newMode });
		onchange?.();
	}
	
	function handleSilenceDurationChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const newDuration = parseInt(target.value, 10);
		localSettings.silenceDuration = newDuration;
		updateSettings({ silenceDuration: newDuration });
		onchange?.();
	}
	
	function handleReset() {
		resetSettings();
		onchange?.();
	}
	
	// Input mode options
	const inputModes: { value: InputMode; label: string; description: string }[] = [
		{ value: 'auto', label: 'Auto-detect', description: 'Automatically detect when you start/stop speaking' },
		{ value: 'push-to-talk', label: 'Push to Talk', description: 'Click button to start, click again to stop' },
		{ value: 'hold-to-talk', label: 'Hold to Talk', description: 'Hold button while speaking' },
	];
	
	// STT language options
	const sttLanguages = [
		{ value: 'en', label: 'English' },
		{ value: 'auto', label: 'Auto-detect' },
		{ value: 'zh', label: 'Chinese' },
		{ value: 'de', label: 'German' },
		{ value: 'es', label: 'Spanish' },
		{ value: 'fr', label: 'French' },
		{ value: 'it', label: 'Italian' },
		{ value: 'ja', label: 'Japanese' },
		{ value: 'ko', label: 'Korean' },
		{ value: 'nl', label: 'Dutch' },
		{ value: 'pl', label: 'Polish' },
		{ value: 'pt', label: 'Portuguese' },
		{ value: 'ru', label: 'Russian' },
	];
	
	function handleLanguageChange(event: Event) {
		const target = event.target as HTMLSelectElement;
		const newLanguage = target.value;
		localSettings.sttLanguage = newLanguage;
		updateSettings({ sttLanguage: newLanguage });
		onchange?.();
	}
</script>

{#if compact}
	<!-- Compact Popover Layout -->
	<div class="w-72 p-4 space-y-4">
		<div class="flex items-center justify-between mb-3">
			<h3 class="font-semibold text-gray-900 dark:text-white text-sm">Voice Settings</h3>
			<button
				type="button"
				onclick={handleReset}
				class="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 flex items-center gap-1"
				title="Reset to defaults"
			>
				<CarbonReset class="w-3.5 h-3.5" />
				Reset
			</button>
		</div>
		
		<!-- Auto TTS Toggle -->
		<label class="flex items-center justify-between cursor-pointer group">
			<div>
				<span class="text-sm font-medium text-gray-900 dark:text-gray-100">Auto TTS</span>
				<p class="text-xs text-gray-500 dark:text-gray-400">Speak responses automatically</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={localSettings.autoTTS}
				onclick={() => handleToggle('autoTTS')}
				class="
					relative inline-flex h-5 w-9 items-center rounded-full transition-colors
					{localSettings.autoTTS ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform
						{localSettings.autoTTS ? 'translate-x-4' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</label>
		
		<!-- Auto Resume Toggle -->
		<label class="flex items-center justify-between cursor-pointer group">
			<div>
				<span class="text-sm font-medium text-gray-900 dark:text-gray-100">Auto Resume</span>
				<p class="text-xs text-gray-500 dark:text-gray-400">Resume listening after speaking</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={localSettings.autoResume}
				onclick={() => handleToggle('autoResume')}
				class="
					relative inline-flex h-5 w-9 items-center rounded-full transition-colors
					{localSettings.autoResume ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform
						{localSettings.autoResume ? 'translate-x-4' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</label>
		
		<!-- RAG Toggle (inverted display - shows "Use RAG" but stores disableRAG) -->
		<label class="flex items-center justify-between cursor-pointer group">
			<div>
				<span class="text-sm font-medium text-gray-900 dark:text-gray-100">Use RAG</span>
				<p class="text-xs text-gray-500 dark:text-gray-400">Include knowledge base context</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={!localSettings.disableRAG}
				onclick={() => handleToggle('disableRAG')}
				class="
					relative inline-flex h-5 w-9 items-center rounded-full transition-colors
					{!localSettings.disableRAG ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform
						{!localSettings.disableRAG ? 'translate-x-4' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</label>
		
		<!-- Input Mode Select -->
		<div>
			<label class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-1.5">
				Input Mode
			</label>
			<select
				value={localSettings.inputMode}
				onchange={handleInputModeChange}
				class="
					w-full rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm
					dark:border-gray-600 dark:bg-gray-700 dark:text-white
					focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500
				"
			>
				{#each inputModes as mode}
					<option value={mode.value}>{mode.label}</option>
				{/each}
			</select>
		</div>
		
		<!-- Silence Duration Slider -->
		<div>
			<label class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-1.5">
				Silence Duration
				<span class="font-normal text-gray-500 dark:text-gray-400 ml-1">
					{localSettings.silenceDuration}ms
				</span>
			</label>
			<input
				type="range"
				min="500"
				max="3000"
				step="100"
				value={localSettings.silenceDuration}
				oninput={handleSilenceDurationChange}
				class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-purple-600"
			/>
			<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
				<span>0.5s</span>
				<span>3s</span>
			</div>
		</div>
		
		<!-- STT Language Select -->
		<div>
			<label class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-1.5">
				Speech Recognition Language
			</label>
			<select
				value={localSettings.sttLanguage}
				onchange={handleLanguageChange}
				class="
					w-full rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm
					dark:border-gray-600 dark:bg-gray-700 dark:text-white
					focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500
				"
			>
				{#each sttLanguages as lang}
					<option value={lang.value}>{lang.label}</option>
				{/each}
			</select>
			<p class="text-xs text-gray-500 dark:text-gray-400 mt-1">Set to English for best accuracy</p>
		</div>
	</div>
{:else}
	<!-- Full Settings Page Layout -->
	<div class="space-y-6">
		<!-- Auto TTS -->
		<div class="flex items-start justify-between p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<div class="flex-1">
				<h4 class="font-medium text-gray-900 dark:text-gray-100">Automatic Text-to-Speech</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
					Automatically play assistant responses as speech when conversation mode is active.
				</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={localSettings.autoTTS}
				onclick={() => handleToggle('autoTTS')}
				class="
					relative inline-flex h-6 w-11 items-center rounded-full transition-colors ml-4
					{localSettings.autoTTS ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform
						{localSettings.autoTTS ? 'translate-x-5' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</div>
		
		<!-- Auto Resume -->
		<div class="flex items-start justify-between p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<div class="flex-1">
				<h4 class="font-medium text-gray-900 dark:text-gray-100">Auto Resume Listening</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
					Automatically resume listening for your voice after the assistant finishes speaking.
				</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={localSettings.autoResume}
				onclick={() => handleToggle('autoResume')}
				class="
					relative inline-flex h-6 w-11 items-center rounded-full transition-colors ml-4
					{localSettings.autoResume ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform
						{localSettings.autoResume ? 'translate-x-5' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</div>
		
		<!-- Use RAG -->
		<div class="flex items-start justify-between p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<div class="flex-1">
				<h4 class="font-medium text-gray-900 dark:text-gray-100">Use Knowledge Base (RAG)</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
					Include relevant documents from your knowledge base when generating responses.
					Disabling this provides faster responses but without document context.
				</p>
			</div>
			<button
				type="button"
				role="switch"
				aria-checked={!localSettings.disableRAG}
				onclick={() => handleToggle('disableRAG')}
				class="
					relative inline-flex h-6 w-11 items-center rounded-full transition-colors ml-4
					{!localSettings.disableRAG ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}
				"
			>
				<span
					class="
						inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform
						{!localSettings.disableRAG ? 'translate-x-5' : 'translate-x-0.5'}
					"
				></span>
			</button>
		</div>
		
		<!-- Input Mode -->
		<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">Voice Input Mode</h4>
			<p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
				Choose how voice input is triggered when conversation mode is active.
			</p>
			<div class="space-y-3">
				{#each inputModes as mode}
					<label class="flex items-start gap-3 cursor-pointer p-3 rounded-lg border border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors {localSettings.inputMode === mode.value ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20' : ''}">
						<input
							type="radio"
							name="inputMode"
							value={mode.value}
							checked={localSettings.inputMode === mode.value}
							onchange={handleInputModeChange}
							class="mt-0.5 accent-purple-600"
						/>
						<div class="flex-1">
							<span class="font-medium text-gray-900 dark:text-gray-100">{mode.label}</span>
							<p class="text-sm text-gray-600 dark:text-gray-400">{mode.description}</p>
						</div>
						{#if localSettings.inputMode === mode.value}
							<CarbonCheckmark class="w-5 h-5 text-purple-600 mt-0.5" />
						{/if}
					</label>
				{/each}
			</div>
		</div>
		
		<!-- STT Language -->
		<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">Speech Recognition Language</h4>
			<p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
				Select the language for speech-to-text transcription. Setting a specific language improves accuracy.
				Use "Auto-detect" only if you speak multiple languages.
			</p>
			<select
				value={localSettings.sttLanguage}
				onchange={handleLanguageChange}
				class="
					w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm
					dark:border-gray-600 dark:bg-gray-700 dark:text-white
					focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500
				"
			>
				{#each sttLanguages as lang}
					<option value={lang.value}>{lang.label}</option>
				{/each}
			</select>
		</div>
		
		<!-- Silence Duration -->
		<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
			<div class="flex items-center justify-between mb-2">
				<h4 class="font-medium text-gray-900 dark:text-gray-100">Silence Duration</h4>
				<span class="text-sm font-mono text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900/30 px-2 py-0.5 rounded">
					{localSettings.silenceDuration}ms
				</span>
			</div>
			<p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
				How long to wait after you stop speaking before automatically sending your message.
				Shorter values feel more responsive, longer values give you more time to think.
			</p>
			<input
				type="range"
				min="500"
				max="3000"
				step="100"
				value={localSettings.silenceDuration}
				oninput={handleSilenceDurationChange}
				class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-purple-600"
			/>
			<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-2">
				<span>Quick (0.5s)</span>
				<span>Normal (1.5s)</span>
				<span>Slow (3s)</span>
			</div>
		</div>
		
		<!-- Reset Button -->
		<div class="flex justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
			<button
				type="button"
				onclick={handleReset}
				class="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
			>
				<CarbonReset class="w-4 h-4" />
				Reset to Defaults
			</button>
		</div>
	</div>
{/if}
