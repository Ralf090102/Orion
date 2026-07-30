import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Orion ships as a Tauri desktop app: the packaged build has no Node
		// server, just static files served over a custom protocol. adapter-static
		// with a fallback produces a single-page-app shell instead of trying to
		// prerender routes like conversation/[id] whose IDs only exist at runtime
		// (see src/routes/+layout.ts: `export const ssr = false`, required for
		// fallback mode since there's no server to run SSR against).
		adapter: adapter({
			pages: 'dist',
			assets: 'dist',
			fallback: 'index.html',
			precompress: false,
			strict: true
		})
	}
};

export default config;
