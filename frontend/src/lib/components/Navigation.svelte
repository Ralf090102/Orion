<script lang="ts">
	import { page } from '$app/stores';
	import { MessageSquare, Upload, Settings, Activity } from 'lucide-svelte';
	
	interface NavItem {
		href: string;
		label: string;
		icon: any;
	}
	
	const navItems: NavItem[] = [
		{ href: '/', label: 'Chat', icon: MessageSquare },
		{ href: '/upload', label: 'Upload', icon: Upload },
		{ href: '/settings', label: 'Settings', icon: Settings }
	];
	
	$: currentPath = $page.url.pathname;
</script>

<nav class="navbar bg-base-200 shadow-lg">
	<div class="navbar-start">
		<div class="dropdown">
			<div tabindex="0" role="button" class="btn btn-ghost btn-circle">
				<svg
					class="w-5 h-5"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
					xmlns="http://www.w3.org/2000/svg">
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M4 6h16M4 12h16M4 18h7"></path>
				</svg>
			</div>
			<ul
				tabindex="-1"
				role="menu"
				class="menu menu-sm dropdown-content bg-base-100 rounded-box z-[1] mt-3 w-52 p-2 shadow">
				{#each navItems as item}
					<li role="none">
						<a href={item.href} class:active={currentPath === item.href} role="menuitem">
							<svelte:component this={item.icon} size={16} />
							{item.label}
						</a>
					</li>
				{/each}
			</ul>
		</div>
	</div>

	<div class="navbar-center">
		<a href="/" class="btn btn-ghost text-xl font-bold text-primary">
			<Activity class="mr-2" size={20} />
			Orion
		</a>
	</div>

	<div class="navbar-end">
		<div class="flex gap-2">
			{#each navItems as item}
				<a 
					href={item.href} 
					class="btn btn-ghost btn-sm hidden sm:flex"
					class:btn-active={currentPath === item.href}>
					<svelte:component this={item.icon} size={16} />
					{item.label}
				</a>
			{/each}
		</div>
	</div>
</nav>
