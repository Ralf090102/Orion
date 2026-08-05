import { UrlDependency } from "$lib/types/UrlDependency";
import { redirect } from "@sveltejs/kit";
import { base } from "$app/paths";
import type { PageLoad } from "./$types";
import { BACKEND_URL } from '$lib/utils/backendUrl';

function generateId(): string {
	if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
		return crypto.randomUUID();
	}
	return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
		const r = (Math.random() * 16) | 0;
		return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
	});
}

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
				id: generateId(),
				from: msg.role === 'user' ? 'user' : 'assistant',
				content: msg.content,
				createdAt: new Date(msg.timestamp),
				updatedAt: new Date(msg.timestamp),
				// The backend now includes sources on reload (previously
				// dropped server-side, see backend/api/chat.py's get_session);
				// map them through so citations survive leaving and
				// re-entering a conversation, not just the live first
				// response. undefined (not []) when empty so ChatMessage's
				// `message.sources?.length` gate behaves the same as a
				// message that never had sources at all.
				sources: msg.sources?.length ? msg.sources : undefined,
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
