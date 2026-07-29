import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import Icons from 'unplugin-icons/vite';

export default defineConfig(({ command }) => ({
	plugins: [
		sveltekit(),
		Icons({
			compiler: 'svelte'
		})
	],
	// Workaround for a SvelteKit dev-server race (sveltejs/kit#13249, #14143):
	// __SVELTEKIT_PAYLOAD__ is sometimes served unsubstituted in early-loaded
	// runtime modules, throwing "ReferenceError: __SVELTEKIT_PAYLOAD__ is not
	// defined". Redeclaring it here (matching the value the kit plugin itself
	// uses in dev) avoids the race. Build/SSR are untouched.
	define: command === 'serve' ? { __SVELTEKIT_PAYLOAD__: 'globalThis.__sveltekit_dev' } : undefined,
	test: {
		expect: { requireAssertions: true },
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					environment: 'browser',
					browser: {
						enabled: true,
						provider: 'playwright',
						instances: [{ browser: 'chromium' }]
					},
					include: ['src/**/*.svelte.{test,spec}.{js,ts}'],
					exclude: ['src/lib/server/**'],
					setupFiles: ['./vitest-setup-client.ts']
				}
			},
			{
				extends: './vite.config.ts',
				test: {
					name: 'server',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
}));
