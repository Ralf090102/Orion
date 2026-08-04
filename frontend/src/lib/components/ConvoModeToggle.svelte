<script lang="ts">
	import { page } from "$app/stores";
	import {
		convoModeState,
		toggleConversationMode,
		initConversationMode,
		type ConvoModeStatus,
	} from "$lib/stores/conversationMode.svelte";
	
	import ConvoModeSettings from "./ConvoModeSettings.svelte";
	
	import MicIcon from "~icons/lucide/mic";
	import MicOffIcon from "~icons/lucide/mic-off";
	import Volume2Icon from "~icons/lucide/volume-2";
	import LoaderIcon from "~icons/lucide/loader-2";
	import SettingsIcon from "~icons/lucide/settings";

	interface Props {
		// Set when there's no active conversation to attach voice mode to
		// (the empty "/" screen). Handles picking/creating a session and
		// navigating there instead of toggling state that has nothing to
		// attach to yet.
		onStartFromEmpty?: () => void | Promise<void>;
	}

	let { onStartFromEmpty }: Props = $props();

	// Settings popover state
	let showSettings = $state(false);
	let settingsRef: HTMLDivElement | null = null;
	
	// Initialize when conversation ID changes
	$effect(() => {
		const conversationId = $page.params.id;
		if (conversationId) {
			initConversationMode(conversationId);
		}
	});
	
	// Close popover when clicking outside
	$effect(() => {
		if (!showSettings) return;
		
		function handleClickOutside(event: MouseEvent) {
			if (settingsRef && !settingsRef.contains(event.target as Node)) {
				showSettings = false;
			}
		}
		
		// Delay adding listener to avoid immediate close
		setTimeout(() => {
			document.addEventListener('click', handleClickOutside);
		}, 0);
		
		return () => {
			document.removeEventListener('click', handleClickOutside);
		};
	});
	
	// Derive from reactive state directly
	const status = $derived(convoModeState.status);
	const enabled = $derived(convoModeState.enabled);
	
	// Status to icon mapping
	const statusConfig: Record<ConvoModeStatus, {
		icon: typeof MicIcon;
		label: string;
		bgClass: string;
		animate: boolean;
	}> = {
		off: {
			icon: MicOffIcon,
			label: "Voice mode off",
			bgClass: "bg-gray-600 hover:bg-gray-500",
			animate: false,
		},
		idle: {
			icon: MicIcon,
			label: "Voice mode on - Click to start listening",
			bgClass: "bg-purple-600 hover:bg-purple-500 shadow-purple-500/40 shadow-lg",
			animate: false,
		},
		listening: {
			icon: MicIcon,
			label: "Listening...",
			bgClass: "bg-red-500 hover:bg-red-400",
			animate: true,
		},
		processing: {
			icon: LoaderIcon,
			label: "Processing...",
			bgClass: "bg-blue-500",
			animate: true,
		},
		speaking: {
			icon: Volume2Icon,
			label: "Speaking...",
			bgClass: "bg-green-500",
			animate: true,
		},
	};
	
	const currentConfig = $derived(statusConfig[status]);
	
	function handleClick() {
		if (status === 'processing' || status === 'speaking') {
			// Don't toggle during active operation
			return;
		}
		if (onStartFromEmpty) {
			onStartFromEmpty();
			return;
		}
		toggleConversationMode();
	}
	
	function handleKeyDown(event: KeyboardEvent) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			handleClick();
		}
	}
	
	function handleSettingsClick(event: MouseEvent) {
		event.stopPropagation();
		showSettings = !showSettings;
	}
</script>

<div class="relative flex items-center gap-1" bind:this={settingsRef}>
	<!-- Main Toggle Button -->
	<button
		type="button"
		onclick={handleClick}
		onkeydown={handleKeyDown}
		class="
			flex items-center justify-center
			w-9 h-9 rounded-full
			transition-all duration-200
			text-white
			{currentConfig.bgClass}
			{currentConfig.animate && status === 'listening' ? 'animate-pulse' : ''}
			{status === 'processing' ? 'cursor-wait' : ''}
			{status === 'speaking' ? 'cursor-default' : 'cursor-pointer'}
		"
		title={currentConfig.label}
		aria-label={currentConfig.label}
		disabled={status === 'processing'}
	>
		{#if status === 'off'}
			<MicOffIcon class="w-5 h-5" />
		{:else if status === 'idle'}
			<MicIcon class="w-5 h-5" />
		{:else if status === 'listening'}
			<MicIcon class="w-5 h-5" />
		{:else if status === 'processing'}
			<LoaderIcon class="w-5 h-5 animate-spin" />
		{:else if status === 'speaking'}
			<Volume2Icon class="w-5 h-5" />
		{/if}
	</button>
	
	<!-- Settings Gear Icon -->
	<button
		type="button"
		onclick={handleSettingsClick}
		class="
			flex items-center justify-center
			w-7 h-7 rounded-full
			transition-all duration-200
			text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200
			hover:bg-gray-200 dark:hover:bg-gray-700
			{showSettings ? 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200' : ''}
		"
		title="Voice settings"
		aria-label="Voice settings"
		aria-expanded={showSettings}
	>
		<SettingsIcon class="w-4 h-4" />
	</button>
	
	<!-- Settings Popover -->
	{#if showSettings}
		<div
			class="
				absolute top-full right-0 mt-2 z-50
				bg-white dark:bg-gray-800
				rounded-xl shadow-lg border border-gray-200 dark:border-gray-700
				animate-in fade-in slide-in-from-top-2 duration-200
			"
		>
			<ConvoModeSettings compact={true} onchange={() => {}} />
		</div>
	{/if}
</div>

<style>
	/* Custom pulse animation for listening state */
	@keyframes pulse-ring {
		0% {
			box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
		}
		70% {
			box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
		}
		100% {
			box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
		}
	}
	
	button:global(.animate-pulse) {
		animation: pulse-ring 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}
</style>
