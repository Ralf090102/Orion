import { redirect } from "@sveltejs/kit";
import { base } from "$app/paths";
import type { PageLoad } from "./$types";

// Simplified share route - just redirect to conversation
export const load: PageLoad = async ({ params }) => {
	// For local use, share routes just redirect to the conversation
	redirect(302, `${base}/conversation/${params.id}`);
};
