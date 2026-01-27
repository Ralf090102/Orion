<script lang="ts">
	export const titles: { [key: string]: string } = {
		today: "Today",
		week: "This week",
		month: "This month",
		older: "Older",
	} as const;
	import { base } from "$app/paths";

	import Logo from "$lib/components/icons/Logo.svelte";
	import IconSun from "$lib/components/icons/IconSun.svelte";
	import IconMoon from "$lib/components/icons/IconMoon.svelte";
	import { switchTheme, subscribeToTheme } from "$lib/switchTheme";
	import { isAborted } from "$lib/stores/isAborted";
	import { onDestroy } from "svelte";

	import NavConversationItem from "./NavConversationItem.svelte";
	import type { LayoutData } from "../../routes/$types";
	import type { ConvSidebar } from "$lib/types/ConvSidebar";
	import type { Model } from "$lib/types/Model";
	import { page } from "$app/state";
	import InfiniteScroll from "./InfiniteScroll.svelte";
	import { CONV_NUM_PER_PAGE } from "$lib/constants/pagination";
	import { browser } from "$app/environment";
	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";
	import { enabledServersCount } from "$lib/stores/mcpServers";
	import { isPro } from "$lib/stores/isPro";
	import IconPro from "$lib/components/icons/IconPro.svelte";
	import MCPServerManager from "./mcp/MCPServerManager.svelte";

	const publicConfig = usePublicConfig();
	const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

	interface Props {
		conversations: ConvSidebar[];
		user: LayoutData["user"];
		p?: number;
		ondeleteConversation?: (id: string) => void;
		oneditConversationTitle?: (payload: { id: string; title: string }) => void;
	}

	let {
		conversations = $bindable(),
		user,
		p = $bindable(0),
		ondeleteConversation,
		oneditConversationTitle,
	}: Props = $props();

	let hasMore = $state(true);

	function handleNewChatClick(e: MouseEvent) {
		isAborted.set(true);
	}

	function handleNavItemClick(e: MouseEvent) {
		// Navigation click handler
	}

	const dateRanges = [
		new Date().setDate(new Date().getDate() - 1),
		new Date().setDate(new Date().getDate() - 7),
		new Date().setMonth(new Date().getMonth() - 1),
	];

	let groupedConversations = $derived({
		today: conversations.filter(({ updatedAt }) => updatedAt.getTime() > dateRanges[0]),
		week: conversations.filter(
			({ updatedAt }) => updatedAt.getTime() > dateRanges[1] && updatedAt.getTime() < dateRanges[0]
		),
		month: conversations.filter(
			({ updatedAt }) => updatedAt.getTime() > dateRanges[2] && updatedAt.getTime() < dateRanges[1]
		),
		older: conversations.filter(({ updatedAt }) => updatedAt.getTime() < dateRanges[2]),
	});

	const nModels: number = page.data.models.filter((el: Model) => !el.unlisted).length;

	async function handleVisible() {
		// Don't fetch if we already know there's no more data
		if (!hasMore) return;
		
		p++;
		try {
			const response = await fetch(`${BACKEND_URL}/api/chat/sessions?limit=${CONV_NUM_PER_PAGE}&offset=${p * CONV_NUM_PER_PAGE}`);
			
			if (!response.ok) {
				console.error('Failed to fetch sessions');
				hasMore = false;
				return;
			}

			const data = await response.json();
			const newConvs: ConvSidebar[] = data.sessions?.map((session: any) => ({
				id: session.session_id,
				title: session.metadata?.title || session.metadata?.topic || 'New Chat',
				updatedAt: new Date(session.updated_at),
				createdAt: new Date(session.created_at),
				model: session.metadata?.model || 'default'
			})) || [];
			
			// If we got fewer results than requested, there's no more data
			if (newConvs.length < CONV_NUM_PER_PAGE) {
				hasMore = false;
			}

			// Filter out duplicates based on session ID
			const existingIds = new Set(conversations.map(c => c.id));
			const uniqueNewConvs = newConvs.filter(conv => !existingIds.has(conv.id));
			
			if (uniqueNewConvs.length === 0) {
				// No new conversations, we've reached the end
				hasMore = false;
			} else {
				conversations = [...conversations, ...uniqueNewConvs];
			}
		} catch (error) {
			console.error('Error fetching conversations:', error);
			hasMore = false;
		}
	}

	$effect(() => {
		if (conversations.length <= CONV_NUM_PER_PAGE) {
			// reset p to 0 if there's only one page of content
			// that would be caused by a data loading invalidation
			p = 0;
		}
	});

	let isDark = $state(false);
	let unsubscribeTheme: (() => void) | undefined;
	let showMcpModal = $state(false);

	if (browser) {
		unsubscribeTheme = subscribeToTheme(({ isDark: nextIsDark }) => {
			isDark = nextIsDark;
		});
	}

	onDestroy(() => {
		unsubscribeTheme?.();
	});
</script>

<!-- Header Section -->
<div
	class="sticky top-0 z-10 flex flex-none touch-none items-center justify-between border-b border-r border-gray-200/80 bg-gray-50/95 px-4 py-3.5 backdrop-blur-sm dark:border-gray-700/80 dark:bg-gray-950/95 max-sm:pt-0"
>
	<a
		class="flex select-none items-center rounded-xl text-lg font-semibold transition-opacity hover:opacity-80"
		href="{publicConfig.PUBLIC_ORIGIN}{base}/"
	>
		<Logo classNames="dark:invert mr-[2px]" />
		{publicConfig.PUBLIC_APP_NAME}
	</a>
	<a
		href={`${base}/`}
		onclick={handleNewChatClick}
		class="flex rounded-lg border border-blue-200 bg-gradient-to-b from-blue-50 to-white px-3 py-1.5 text-center text-sm font-medium text-blue-700 shadow-sm transition-all hover:shadow-md hover:border-blue-300 dark:border-blue-800 dark:from-blue-950 dark:to-gray-800 dark:text-blue-300 dark:hover:border-blue-700"
		title="Ctrl/Cmd + Shift + O"
	>
		New Chat
	</a>
</div>

<!-- Conversations Section -->
<div
	class="scrollbar-custom flex flex-1 touch-pan-y flex-col gap-1 overflow-y-auto border-r border-gray-200/80 bg-gradient-to-br from-gray-100 via-gray-50 to-gray-100 px-3 pb-3 pt-3 text-[.9rem] dark:border-gray-700/80 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950"
>
	<div class="flex flex-col gap-0.5">
		{#each Object.entries(groupedConversations) as [group, convs]}
			{#if convs.length}
				<div class="mb-2 mt-3 first:mt-0">
					<h4 class="mb-2 pl-2 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
						{titles[group]}
					</h4>
					<div class="flex flex-col gap-0.5">
						{#each convs as conv}
							<NavConversationItem {conv} {oneditConversationTitle} {ondeleteConversation} />
						{/each}
					</div>
				</div>
			{/if}
		{/each}
	</div>
	{#if hasMore}
		<InfiniteScroll onvisible={handleVisible} />
	{/if}
</div>

<!-- Bottom Navigation Section -->
<div
	class="flex touch-none flex-col gap-2 border-t border-r border-gray-200/80 bg-gray-50/95 p-3 text-sm backdrop-blur-sm dark:border-gray-700/80 dark:bg-gray-950/95"
>
	<!-- Models Link -->
	<a
		href="{base}/models"
		class="group flex h-10 flex-none items-center gap-2 rounded-lg border border-transparent bg-white/90 px-3 text-gray-700 shadow-sm transition-all hover:border-gray-300 hover:bg-white hover:shadow-md dark:bg-gray-900/80 dark:text-gray-300 dark:hover:border-gray-600 dark:hover:bg-gray-800"
		onclick={handleNavItemClick}
	>
		<span class="font-medium">Models</span>
		<span
			class="ml-auto rounded-md bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-700 transition-colors group-hover:bg-blue-200 dark:bg-blue-900/50 dark:text-blue-300 dark:group-hover:bg-blue-900"
			>{nModels}</span
		>
	</a>

	<!-- Settings and Theme Row -->
	<div class="flex gap-2">
		<a
			href="{base}/settings/application"
			class="flex h-10 flex-1 items-center gap-2 rounded-lg border border-transparent bg-white/90 px-3 font-medium text-gray-700 shadow-sm transition-all hover:border-gray-300 hover:bg-white hover:shadow-md dark:bg-gray-900/80 dark:text-gray-300 dark:hover:border-gray-600 dark:hover:bg-gray-800"
			onclick={handleNavItemClick}
		>
			Settings
		</a>
		<button
			onclick={() => {
				switchTheme();
			}}
			aria-label="Toggle theme"
			class="flex size-10 flex-none items-center justify-center rounded-lg border border-transparent bg-white/90 p-2 text-gray-700 shadow-sm transition-all hover:border-gray-300 hover:bg-white hover:shadow-md dark:bg-gray-900/80 dark:text-gray-300 dark:hover:border-gray-600 dark:hover:bg-gray-800"
		>
			{#if browser}
				{#if isDark}
					<IconSun />
				{:else}
					<IconMoon />
				{/if}
			{/if}
		</button>
	</div>
</div>

{#if showMcpModal}
	<MCPServerManager onclose={() => (showMcpModal = false)} />
{/if}
