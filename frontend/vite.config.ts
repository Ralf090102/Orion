import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import Icons from 'unplugin-icons/vite';

export default defineConfig({
	plugins: [
		sveltekit(),
		Icons({
			compiler: 'svelte'
		})
	],
	// A __SVELTEKIT_PAYLOAD__ is not defined workaround used to live here as
	// a `define` text-substitution (sveltejs/kit#13249, #14143) -- but
	// upstream's own diagnosis of #13249 is that `define` doesn't reliably
	// rewrite that reference for code evaluated early in the dev-server's
	// module graph, which matched what we saw: it only avoided the crash on
	// a fraction of fresh page loads. Replaced with a real global declared
	// in src/app.html's first <script>, which sidesteps the unreliable
	// mechanism instead of leaning on it further -- see the comment there.
	server: {
		watch: {
			// src-tauri/ is a Rust project nested inside this one; Vite's
			// project root is `frontend/`, so its watcher picks it up by
			// default. That's mostly harmless -- until `tauri-build`'s
			// build script stages `bundle.resources` (tauri.conf.json) into
			// src-tauri/target/<profile>/ on every `cargo build`/`cargo run`
			// (dev included, not just release), which for Orion means the
			// full ~2GB/56,000-file portable Python runtime. Vite's watcher
			// fired thousands of spurious "page reload" events processing
			// that during `npx tauri dev`'s build, landing right when the
			// SvelteKit client hydrates -- exactly the kind of timing
			// pressure that turns the rare race above into a near-certain
			// blank first load. Rust source changes are already watched and
			// rebuilt by Tauri's own `cargo` process, so Vite never needed
			// to watch this directory at all.
			ignored: ['**/src-tauri/**']
		}
	},
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
});
