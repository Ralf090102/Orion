// Simplified hooks for local use - no authentication or server-side database
import type { Handle, HandleServerError } from "@sveltejs/kit";
import { dev } from "$app/environment";

// Simple error handler
export const handleError: HandleServerError = async ({ error, event, status, message }) => {
	// Handle 404s
	if (event.route.id === null) {
		return {
			message: `Page ${event.url.pathname} not found`,
		};
	}

	const errorId = crypto.randomUUID();

	// Log error in development
	if (dev) {
		console.error({
			url: event.request.url,
			params: event.params,
			message,
			error,
			errorId,
			status,
		});
	}

	return {
		message: dev ? (error instanceof Error ? error.message : String(error)) : "An error occurred",
		errorId,
	};
};

// Simple request handler - no authentication needed for local use
export const handle: Handle = async ({ event, resolve }) => {
	// Just resolve the request normally
	return resolve(event);
};
