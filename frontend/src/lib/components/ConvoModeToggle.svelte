<script lang="ts">
	import { page } from "$app/stores";
	import {
		conversationModeState,
		toggleConversationMode,
		initConversationMode,
		type ConvoModeStatus,
	} from "$lib/stores/conversationMode.svelte";
	
	import MicIcon from "~icons/lucide/mic";
	import MicOffIcon from "~icons/lucide/mic-off";
	import Volume2Icon from "~icons/lucide/volume-2";
	import LoaderIcon from "~icons/lucide/loader-2";
	
	// Initialize when conversation ID changes
	$effect(() => {
		const conversationId = $page.params.id;
		if (conversationId) {
			initConversationMode(conversationId);
		}
	});
	
	// Get reactive state
	const status = $derived(conversationModeState.status);
	const enabled = $derived(conversationModeState.enabled);
	
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
		toggleConversationMode();
	}
	
	function handleKeyDown(event: KeyboardEvent) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			handleClick();
		}
	}
</script>

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
