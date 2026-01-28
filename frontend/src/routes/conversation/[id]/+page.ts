import { UrlDependency } from "$lib/types/UrlDependency";
import { redirect } from "@sveltejs/kit";
import { base } from "$app/paths";
import type { PageLoad } from "./$types";

const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

export const load: PageLoad = async ({ params, depends, fetch, parent }) => {
	depends(UrlDependency.Conversation);

	const parentData = await parent();

	// Load conversation from FastAPI backend
	try {
		const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${params.id}`);
		
		if (!response.ok) {
			throw new Error('Session not found');
		}

		const sessionData = await response.json();

		// Convert to expected format
		return {
			messages: sessionData.messages?.map((msg: any) => ({
				id: crypto.randomUUID(),
				from: msg.role === 'user' ? 'user' : 'assistant',
				content: msg.content,
				createdAt: new Date(msg.timestamp),
				updatedAt: new Date(msg.timestamp),
			})) || [],
			conversations: parentData.conversations,
			models: parentData.models,
			oldModels: parentData.oldModels || [],
			model: parentData.models[0]?.id || 'default',
			title: sessionData.metadata?.title || sessionData.metadata?.topic || 'Chat',
			preprompt: '',
			rootMessageId: null,
			shared: false,
			transcriptionEnabled: true, // Enable microphone button
		};
	} catch (err) {
		console.error('Failed to load conversation:', err);
		redirect(302, `${base}/`);
	}
};
