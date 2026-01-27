<script lang="ts">
	import { base } from "$app/paths";
	import { page } from "$app/state";

	import CarbonTrashCan from "~icons/carbon/trash-can";
	import CarbonEdit from "~icons/carbon/edit";
	import type { ConvSidebar } from "$lib/types/ConvSidebar";

	import EditConversationModal from "$lib/components/EditConversationModal.svelte";
	import DeleteConversationModal from "$lib/components/DeleteConversationModal.svelte";
	import { requireAuthUser } from "$lib/utils/auth";

	interface Props {
		conv: ConvSidebar;
		readOnly?: true;
		ondeleteConversation?: (id: string) => void;
		oneditConversationTitle?: (payload: { id: string; title: string }) => void;
	}

	let { conv, readOnly, ondeleteConversation, oneditConversationTitle }: Props = $props();

	let deleteOpen = $state(false);
	let renameOpen = $state(false);
</script>

<a
	data-sveltekit-noscroll
	data-sveltekit-preload-data="tap"
	href="{base}/conversation/{conv.id}"
	class="group flex h-9 flex-none items-center gap-2 rounded-lg border border-transparent px-3 text-gray-700 transition-all hover:border-gray-300 hover:bg-white hover:shadow-sm dark:text-gray-300 dark:hover:border-gray-600 dark:hover:bg-gray-800
		{conv.id === page.params.id 
			? 'border-blue-200 bg-gradient-to-r from-blue-50 to-white shadow-sm dark:border-blue-700 dark:from-blue-950/70 dark:to-blue-900/40' 
			: 'bg-white/60 dark:bg-gray-800/50'}"
>
	<div class="min-w-0 flex-1 truncate first-letter:uppercase">
		<span class="font-medium {conv.id === page.params.id ? 'text-blue-700 dark:text-blue-300' : ''}">{conv.title}</span>
	</div>

	{#if !readOnly}
		<button
			type="button"
			class="flex size-6 items-center justify-center rounded transition-colors md:opacity-0 md:group-hover:opacity-100 hover:bg-gray-200 dark:hover:bg-gray-600"
			title="Edit conversation title"
			onclick={(e) => {
				e.preventDefault();
				if (requireAuthUser()) return;
				renameOpen = true;
			}}
		>
			<CarbonEdit class="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
		</button>

		<button
			type="button"
			class="flex size-6 items-center justify-center rounded transition-colors md:opacity-0 md:group-hover:opacity-100 hover:bg-red-100 dark:hover:bg-red-900/30"
			title="Delete conversation"
			onclick={(event) => {
				event.preventDefault();
				if (requireAuthUser()) return;
				if (event.shiftKey) {
					ondeleteConversation?.(conv.id.toString());
				} else {
					deleteOpen = true;
				}
			}}
		>
			<CarbonTrashCan class="text-sm text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400" />
		</button>
	{/if}
</a>

<!-- Edit title modal -->
{#if renameOpen}
	<EditConversationModal
		open={renameOpen}
		title={conv.title}
		onclose={() => (renameOpen = false)}
		onsave={(payload) => {
			renameOpen = false;
			oneditConversationTitle?.({ id: conv.id.toString(), title: payload.title });
		}}
	/>
{/if}

<!-- Delete confirmation modal -->
{#if deleteOpen}
	<DeleteConversationModal
		open={deleteOpen}
		title={conv.title}
		onclose={() => (deleteOpen = false)}
		ondelete={() => {
			deleteOpen = false;
			ondeleteConversation?.(conv.id.toString());
		}}
	/>
{/if}
