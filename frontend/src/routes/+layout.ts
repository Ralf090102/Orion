import { UrlDependency } from "$lib/types/UrlDependency";
import type { ConvSidebar } from "$lib/types/ConvSidebar";
import { BACKEND_URL } from '$lib/utils/backendUrl';
import { browser } from '$app/environment';

// The Tauri production build has no Node server behind it (just static files
// served over a custom protocol), so SSR/prerendering can never run there --
// this makes the whole app a client-rendered SPA, required for adapter-static's
// fallback mode (see svelte.config.js) and for dynamic routes like
// conversation/[id] whose IDs only exist at runtime, not at build time.
export const ssr = false;

export const load = async ({ depends, fetch }) => {
	depends(UrlDependency.ConversationList);

	// Only fetch from backend in browser context (not during SSR)
	// During Tauri dev, SSR happens before Python backend is available
	const shouldFetch = browser;

	// Load conversations from FastAPI backend
	let conversations: ConvSidebar[] = [];
	if (shouldFetch) {
		try {
			const response = await fetch(`${BACKEND_URL}/api/chat/sessions`);
			if (response.ok) {
				const data = await response.json();
				conversations = data.sessions?.map((session: any) => ({
					id: session.session_id,
					title: session.metadata?.title || session.metadata?.topic || 'New Chat',
					model: session.metadata?.model || 'default',
					updatedAt: new Date(session.updated_at),
					createdAt: new Date(session.created_at),
				})) || [];
			}
		} catch (err) {
			console.error('Failed to load conversations:', err);
		}
	}

	// Fetch actual active model from backend
	let activeModelName = 'mistral:latest';
	if (shouldFetch) {
		try {
			const response = await fetch(`${BACKEND_URL}/api/models/config`);
			if (response.ok) {
				const config = await response.json();
				activeModelName = config.model;
			}
		} catch (err) {
			console.error('Failed to load model config:', err);
		}
	}

	// Create model entry based on active model
	const models = [
		{
			id: activeModelName,
			name: activeModelName,
			displayName: activeModelName,
			description: 'Ollama model',
			websiteUrl: '',
			modelUrl: '',
			datasetName: '',
			datasetUrl: '',
			preprompt: '',
			chatPromptTemplate: '',
			parameters: {
				temperature: 0.7,
				top_p: 0.95,
				max_new_tokens: 2048,
			},
		},
	];

	// Mock public config
	const publicConfig = {
		PUBLIC_APP_NAME: import.meta.env.PUBLIC_APP_NAME || 'Orion',
		PUBLIC_APP_DESCRIPTION: import.meta.env.PUBLIC_APP_DESCRIPTION || 'Local RAG Chat',
		PUBLIC_ORIGIN: import.meta.env.PUBLIC_ORIGIN || 'http://localhost:5173',
		isHuggingChat: false,
		assetPath: '/chatui',
		PUBLIC_PLAUSIBLE_SCRIPT_URL: undefined,
		PUBLIC_APPLE_APP_ID: undefined,
	};

	// Settings with actual active model
	const settings = {
		activeModel: activeModelName,
		customPrompts: {},
		hidePromptExamples: {},
		multimodalOverrides: {},
		toolsOverrides: {},
		welcomeModalSeen: true,
		welcomeModalSeenAt: null,
		directPaste: false,
		disableStream: false,
		shareConversationsWithModelAuthors: false,
	};

	return {
		conversations,
		models,
		oldModels: [],
		user: null, // No authentication for local use
		settings,
		publicConfig,
		transcriptionEnabled: true, // Enable microphone button
		tools: [],
	};
};
