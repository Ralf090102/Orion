/**
 * Chat API Utilities
 * 
 * Functions for interacting with the conversation branching API endpoints.
 * Handles message deletion, branching, and switching for retry/edit functionality.
 */

const BACKEND_URL = import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000';

export interface DeleteMessageResponse {
	status: string;
	message: string;
	deleted_count: number;
}

export interface BranchResponse {
	status: string;
	message: string;
	message_id: string;
	parent_id: string;
}

export interface BranchInfo {
	message_id: string;
	role: string;
	content: string;
	tokens: number;
	timestamp: string;
	is_active: boolean;
}

export interface BranchesResponse {
	status: string;
	parent_id: string | null;
	branches: BranchInfo[];
	total: number;
}

export interface SwitchBranchResponse {
	status: string;
	message: string;
	message_id: string;
}

/**
 * Delete a message and all its children (cascading delete).
 * Used for retry/edit to remove old attempts.
 */
export async function deleteMessage(
	sessionId: string,
	messageId: string
): Promise<DeleteMessageResponse> {
	const response = await fetch(
		`${BACKEND_URL}/api/chat/sessions/${sessionId}/messages/${messageId}`,
		{
			method: 'DELETE',
			headers: {
				'Content-Type': 'application/json',
			},
		}
	);

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to delete message');
	}

	return response.json();
}

/**
 * Create a new conversation branch from a parent message.
 * Used for creating alternative responses while preserving conversation tree.
 */
export async function createBranch(
	sessionId: string,
	parentId: string,
	role: 'user' | 'assistant',
	content: string,
	tokens: number = 0,
	deactivateSiblings: boolean = true
): Promise<BranchResponse> {
	const response = await fetch(
		`${BACKEND_URL}/api/chat/sessions/${sessionId}/branches`,
		{
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				parent_id: parentId,
				role,
				content,
				tokens,
				deactivate_siblings: deactivateSiblings,
			}),
		}
	);

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to create branch');
	}

	return response.json();
}

/**
 * Get all alternative branches from a parent message.
 * Used to display "2 other responses" UI.
 */
export async function getBranches(
	sessionId: string,
	parentId?: string | null
): Promise<BranchesResponse> {
	const url = new URL(`${BACKEND_URL}/api/chat/sessions/${sessionId}/branches`);
	if (parentId) {
		url.searchParams.set('parent_id', parentId);
	}

	const response = await fetch(url.toString(), {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
		},
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to get branches');
	}

	return response.json();
}

/**
 * Switch to a different conversation branch.
 * Activates the specified message and deactivates siblings.
 */
export async function switchBranch(
	sessionId: string,
	messageId: string
): Promise<SwitchBranchResponse> {
	const response = await fetch(
		`${BACKEND_URL}/api/chat/sessions/${sessionId}/branches/switch`,
		{
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				message_id: messageId,
			}),
		}
	);

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to switch branch');
	}

	return response.json();
}

/**
 * Get only messages from the active conversation branch.
 * Returns a clean linear conversation history.
 */
export async function getActiveMessages(sessionId: string): Promise<any> {
	const response = await fetch(
		`${BACKEND_URL}/api/chat/sessions/${sessionId}/messages/active`,
		{
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
			},
		}
	);

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to get active messages');
	}

	return response.json();
}

/**
 * Helper: Retry flow - delete old response and re-send same message
 */
export async function retryMessage(
	sessionId: string,
	assistantMessageId: string
): Promise<void> {
	// Delete the assistant's response
	await deleteMessage(sessionId, assistantMessageId);
	// Caller should then re-send the user message to get a new response
}

/**
 * Helper: Edit flow - delete old message + response and send new content
 */
export async function editMessage(
	sessionId: string,
	userMessageId: string
): Promise<void> {
	// Delete the user message (and its assistant response via CASCADE)
	await deleteMessage(sessionId, userMessageId);
	// Caller should then send the edited message content
}
