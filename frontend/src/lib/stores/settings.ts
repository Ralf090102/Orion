import { browser } from "$app/environment";
import { invalidate } from "$app/navigation";
import { UrlDependency } from "$lib/types/UrlDependency";
import { getContext, setContext } from "svelte";
import { type Writable, writable, get } from "svelte/store";

type SettingsStore = {
	shareConversationsWithModelAuthors: boolean;
	welcomeModalSeen: boolean;
	welcomeModalSeenAt: Date | null;
	activeModel: string;
	customPrompts: Record<string, string>;
	multimodalOverrides: Record<string, boolean>;
	toolsOverrides: Record<string, boolean>;
	recentlySaved: boolean;
	disableStream: boolean;
	directPaste: boolean;
	hidePromptExamples: Record<string, boolean>;
	billingOrganization?: string;
};

type SettingsStoreWritable = Writable<SettingsStore> & {
	instantSet: (settings: Partial<SettingsStore>) => Promise<void>;
	initValue: <K extends keyof SettingsStore>(
		key: K,
		nestedKey: string,
		value: string | boolean
	) => Promise<void>;
};

export function useSettingsStore() {
	return getContext<SettingsStoreWritable>("settings");
}

export function createSettingsStore(initialValue: Omit<SettingsStore, "recentlySaved">) {
	const baseStore = writable({ ...initialValue, recentlySaved: false });

	let timeoutId: NodeJS.Timeout;
	let showSavedOnNextSync = false;

	async function setSettings(settings: Partial<SettingsStore>) {
		baseStore.update((s) => ({
			...s,
			...settings,
		}));

		if (browser) {
			showSavedOnNextSync = true; // User edit, should show "Saved"
			clearTimeout(timeoutId);
			timeoutId = setTimeout(() => {
				invalidate(UrlDependency.ConversationList);

				if (showSavedOnNextSync) {
					// set savedRecently to true for 3s
					baseStore.update((s) => ({
						...s,
						recentlySaved: true,
					}));
					setTimeout(() => {
						baseStore.update((s) => ({
							...s,
							recentlySaved: false,
						}));
					}, 3000);
				}

				showSavedOnNextSync = false;
			}, 300);
			// debounce by 300ms -- settings persist client-side (this store's
			// subscribers write to localStorage); nothing server-side to sync to
			// in a Tauri desktop build, which has no Node server to run SvelteKit
			// +server.ts routes against.
		}
	}

	async function initValue<K extends keyof SettingsStore>(
		key: K,
		nestedKey: string,
		value: string | boolean
	) {
		const currentStore = get(baseStore);
		const currentNestedObject = currentStore[key] as Record<string, string | boolean>;

		// Only initialize if undefined
		if (currentNestedObject?.[nestedKey] !== undefined) {
			return;
		}

		// Update the store
		const newNestedObject = {
			...(currentNestedObject || {}),
			[nestedKey]: value,
		};

		baseStore.update((s) => ({
			...s,
			[key]: newNestedObject,
		}));

		// Persist (debounced) - note: we don't set showSavedOnNextSync
		if (browser) {
			clearTimeout(timeoutId);
			timeoutId = setTimeout(() => {
				invalidate(UrlDependency.ConversationList);

				if (showSavedOnNextSync) {
					baseStore.update((s) => ({
						...s,
						recentlySaved: true,
					}));
					setTimeout(() => {
						baseStore.update((s) => ({
							...s,
							recentlySaved: false,
						}));
					}, 3000);
				}

				showSavedOnNextSync = false;
			}, 300);
		}
	}
	async function instantSet(settings: Partial<SettingsStore>) {
		baseStore.update((s) => ({
			...s,
			...settings,
		}));

		if (browser) {
			invalidate(UrlDependency.ConversationList);
		}
	}

	const newStore = {
		subscribe: baseStore.subscribe,
		set: setSettings,
		instantSet,
		initValue,
		update: (fn: (s: SettingsStore) => SettingsStore) => {
			setSettings(fn(get(baseStore)));
		},
	} satisfies SettingsStoreWritable;

	setContext("settings", newStore);

	return newStore;
}
