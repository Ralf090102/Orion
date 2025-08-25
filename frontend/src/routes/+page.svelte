<script lang="ts">
	import { onMount } from 'svelte';
	import Navigation from '$lib/components/Navigation.svelte';
	import { chatMessages, currentQuery, chatStore } from '$lib/stores/chat';
	import { queryApi } from '$lib/utils/api';
	import { Send, Loader2 } from 'lucide-svelte';
	
	let isLoading = false;
	let chatContainer: HTMLElement;
	
	const handleSubmit = async () => {
		if (!$currentQuery.trim() || isLoading) return;
		
		const userQuery = $currentQuery.trim();
		currentQuery.set('');
		
		// Add user message
		chatStore.addMessage({
			role: 'user',
			content: userQuery
		});
		
		isLoading = true;
		
		try {
			// Scroll to bottom
			setTimeout(() => scrollToBottom(), 100);
			
			const response = await queryApi.query({
				query: userQuery,
				include_sources: true,
				max_results: 5
			});
			
			// Add assistant response
			chatStore.addMessage({
				role: 'assistant',
				content: response.answer,
				sources: response.sources
			});
			
		} catch (error) {
			console.error('Query failed:', error);
			chatStore.addMessage({
				role: 'assistant',
				content: 'Sorry, I encountered an error processing your request. Please try again.'
			});
		} finally {
			isLoading = false;
			setTimeout(() => scrollToBottom(), 100);
		}
	};
	
	const scrollToBottom = () => {
		if (chatContainer) {
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	};
	
	const handleKeydown = (event: KeyboardEvent) => {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSubmit();
		}
	};
</script>

<Navigation />

<div class="container mx-auto max-w-4xl p-4 h-[calc(100vh-4rem)] flex flex-col">
	<div class="flex-1 flex flex-col">
		<!-- Chat Messages -->
		<div 
			bind:this={chatContainer}
			class="flex-1 overflow-y-auto p-4 space-y-4">
			
			{#if $chatMessages.length === 0}
				<div class="text-center py-12">
					<div class="text-4xl mb-4">🤖</div>
					<h2 class="text-2xl font-semibold mb-2">Welcome to Orion</h2>
					<p class="text-base-content/70">
						Your personal RAG assistant is ready to help!<br>
						Ask me anything about your uploaded documents.
					</p>
				</div>
			{:else}
				{#each $chatMessages as message}
					<div class="chat" class:chat-end={message.role === 'user'} class:chat-start={message.role === 'assistant'}>
						<div class="chat-image avatar">
							<div class="w-10 rounded-full bg-base-300 flex items-center justify-center">
								{#if message.role === 'user'}
									<span class="text-xs font-bold">You</span>
								{:else}
									<span class="text-xs font-bold">🤖</span>
								{/if}
							</div>
						</div>
						
						<div class="chat-bubble" class:chat-bubble-primary={message.role === 'user'}>
							<div class="prose max-w-none">
								{message.content}
							</div>
							
							{#if message.sources && message.sources.length > 0}
								<div class="mt-2 pt-2 border-t border-base-content/20">
									<div class="text-xs font-semibold mb-1">Sources:</div>
									{#each message.sources as source}
										<div class="text-xs bg-base-100/20 p-1 rounded mb-1">
											📄 {source.filename} (Score: {source.relevance_score.toFixed(2)})
										</div>
									{/each}
								</div>
							{/if}
						</div>
						
						<div class="chat-footer opacity-50 text-xs">
							{message.timestamp.toLocaleTimeString()}
						</div>
					</div>
				{/each}
				
				{#if isLoading}
					<div class="chat chat-start">
						<div class="chat-image avatar">
							<div class="w-10 rounded-full bg-base-300 flex items-center justify-center">
								<Loader2 size={16} class="animate-spin" />
							</div>
						</div>
						<div class="chat-bubble">
							<div class="flex items-center gap-2">
								<span class="loading loading-dots loading-sm"></span>
								Thinking...
							</div>
						</div>
					</div>
				{/if}
			{/if}
		</div>
		
		<!-- Input Area -->
		<div class="p-4 border-t border-base-300">
			<div class="flex gap-2">
				<textarea
					bind:value={$currentQuery}
					on:keydown={handleKeydown}
					placeholder="Ask me anything about your documents..."
					class="textarea textarea-bordered flex-1 min-h-[2.5rem] max-h-32 resize-none"
					rows="1"
					disabled={isLoading}></textarea>
				
				<button
					on:click={handleSubmit}
					disabled={!$currentQuery.trim() || isLoading}
					class="btn btn-primary">
					{#if isLoading}
						<Loader2 size={16} class="animate-spin" />
					{:else}
						<Send size={16} />
					{/if}
				</button>
			</div>
		</div>
	</div>
</div>
