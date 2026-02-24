<script lang="ts">
	import { page } from "$app/stores";
	import { goto } from "$app/navigation";
	import CarbonArrowLeft from "~icons/carbon/arrow-left";
	import CarbonMicrophone from "~icons/carbon/microphone";
	import CarbonInformation from "~icons/carbon/information";
	
	import ConvoModeSettings from "$lib/components/ConvoModeSettings.svelte";
	import { initConversationMode, convoModeState } from "$lib/stores/conversationMode.svelte";
	
	// Check if we came from a conversation (for back button)
	let returnPath = $state<string | null>(null);
	
	$effect(() => {
		// Try to get the referrer from sessionStorage or use a default
		const stored = sessionStorage.getItem('settings-return-path');
		if (stored) {
			returnPath = stored;
		}
	});
	
	function handleBack() {
		if (returnPath) {
			goto(returnPath);
		} else {
			goto('/settings/application');
		}
	}
	
	// Note: We're showing global/default settings here
	// Per-conversation settings are managed via the quick settings popover in the chat header
	// This page can be used to set defaults or configure when no active conversation
</script>

<svelte:head>
	<title>Conversation Mode Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<!-- Header -->
	<div class="flex items-center gap-4">
		<button
			type="button"
			onclick={handleBack}
			class="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
			title="Back to settings"
		>
			<CarbonArrowLeft class="w-5 h-5 text-gray-600 dark:text-gray-400" />
		</button>
		<div class="flex items-center gap-3">
			<div class="rounded-lg bg-purple-100 p-2 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
				<CarbonMicrophone class="w-6 h-6" />
			</div>
			<div>
				<h1 class="text-2xl font-bold text-gray-900 dark:text-white">Conversation Mode</h1>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					Configure voice conversation settings
				</p>
			</div>
		</div>
	</div>
	
	<!-- Info Banner -->
	<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
		<div class="flex items-start gap-3">
			<CarbonInformation class="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
			<div class="flex-1">
				<h3 class="font-semibold text-blue-900 dark:text-blue-200 text-sm">
					Per-Conversation Settings
				</h3>
				<p class="text-sm text-blue-800 dark:text-blue-300 mt-1">
					Settings on this page affect new conversations. Each conversation stores its own settings
					which can be adjusted using the gear icon next to the conversation mode toggle in the chat header.
				</p>
			</div>
		</div>
	</div>
	
	<!-- Settings Section -->
	<div class="max-w-2xl">
		<h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
			Default Settings
		</h2>
		
		<ConvoModeSettings compact={false} />
	</div>
	
	<!-- Feature Overview -->
	<div class="max-w-2xl mt-4">
		<h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
			How Conversation Mode Works
		</h2>
		
		<div class="space-y-4">
			<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
				<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">1. Voice Detection</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					When conversation mode is enabled, Orion uses Voice Activity Detection (VAD) to automatically
					detect when you start and stop speaking. The sensitivity can be adjusted with the silence duration setting.
				</p>
			</div>
			
			<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
				<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">2. Speech-to-Text</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					Your speech is transcribed using Whisper, a state-of-the-art speech recognition model.
					The transcribed text is then sent to the LLM for processing.
				</p>
			</div>
			
			<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
				<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">3. Brief Responses</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					In conversation mode, the LLM is instructed to give brief, TTS-friendly responses.
					This makes the conversation feel more natural and reduces wait time.
				</p>
			</div>
			
			<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
				<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">4. Text-to-Speech</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					When Auto TTS is enabled, responses are automatically spoken aloud using streaming
					text-to-speech synthesis. You can interrupt at any time by starting to speak.
				</p>
			</div>
			
			<div class="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
				<h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">5. Continuous Loop</h4>
				<p class="text-sm text-gray-600 dark:text-gray-400">
					With Auto Resume enabled, Orion will start listening again after finishing speaking,
					creating a natural back-and-forth conversation flow.
				</p>
			</div>
		</div>
	</div>
	
	<!-- Status Indicators Legend -->
	<div class="max-w-2xl mt-4">
		<h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
			Status Indicators
		</h2>
		
		<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
			<div class="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
				<div class="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
					<span class="text-white text-xs">🎙️</span>
				</div>
				<div>
					<p class="font-medium text-gray-900 dark:text-gray-100 text-sm">Off</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">Voice mode disabled</p>
				</div>
			</div>
			
			<div class="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
				<div class="w-8 h-8 rounded-full bg-purple-600 shadow-lg shadow-purple-500/40 flex items-center justify-center">
					<span class="text-white text-xs">🎙️</span>
				</div>
				<div>
					<p class="font-medium text-gray-900 dark:text-gray-100 text-sm">Idle</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">Ready, waiting for speech</p>
				</div>
			</div>
			
			<div class="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
				<div class="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center animate-pulse">
					<span class="text-white text-xs">●</span>
				</div>
				<div>
					<p class="font-medium text-gray-900 dark:text-gray-100 text-sm">Listening</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">Recording your speech</p>
				</div>
			</div>
			
			<div class="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
				<div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
					<span class="text-white text-xs">⟳</span>
				</div>
				<div>
					<p class="font-medium text-gray-900 dark:text-gray-100 text-sm">Processing</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">Transcribing & generating</p>
				</div>
			</div>
			
			<div class="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
				<div class="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
					<span class="text-white text-xs">🔊</span>
				</div>
				<div>
					<p class="font-medium text-gray-900 dark:text-gray-100 text-sm">Speaking</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">Playing TTS response</p>
				</div>
			</div>
		</div>
	</div>
</div>
