// frontend/src/lib/tauri.ts
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { sendNotification, isPermissionGranted, requestPermission } from '@tauri-apps/plugin-notification';

/**
 * Check if the app is running in Tauri (desktop mode)
 */
export function isTauri(): boolean {
	return typeof window !== 'undefined' && '__TAURI__' in window;
}

/**
 * Get the status of the Python backend
 * @returns "running" | "stopped" | "error" | "browser-mode"
 */
export async function getBackendStatus(): Promise<string> {
	if (!isTauri()) {
		return 'browser-mode';
	}

	try {
		return await invoke<string>('get_backend_status');
	} catch (error) {
		console.error('Failed to get backend status:', error);
		return 'error';
	}
}

/**
 * Restart the Python backend
 */
export async function restartBackend(): Promise<void> {
	if (!isTauri()) {
		console.warn('Backend restart not available in browser mode');
		return;
	}

	try {
		await invoke('restart_backend');
	} catch (error) {
		console.error('Failed to restart backend:', error);
		throw error;
	}
}

/**
 * Stop the Python backend
 */
export async function stopBackend(): Promise<void> {
	if (!isTauri()) {
		console.warn('Backend stop not available in browser mode');
		return;
	}

	try {
		await invoke('stop_backend');
	} catch (error) {
		console.error('Failed to stop backend:', error);
		throw error;
	}
}

/**
 * Start the Python backend
 */
export async function startBackend(): Promise<void> {
	if (!isTauri()) {
		console.warn('Backend start not available in browser mode');
		return;
	}

	try {
		await invoke('start_backend');
	} catch (error) {
		console.error('Failed to start backend:', error);
		throw error;
	}
}

/**
 * Open a folder in the system file explorer
 * @param path - Path to the folder to open
 */
export async function openFolder(path: string): Promise<void> {
	if (!isTauri()) {
		console.warn('Open folder not available in browser mode');
		return;
	}

	try {
		await invoke('open_folder', { path });
	} catch (error) {
		console.error('Failed to open folder:', error);
		throw error;
	}
}

/**
 * Poll backend status periodically
 * @param callback - Function to call with the status
 * @param intervalMs - Polling interval in milliseconds (default: 5000)
 * @returns Cleanup function to stop polling
 */
export function pollBackendStatus(
	callback: (status: string) => void,
	intervalMs: number = 5000
): () => void {
	if (!isTauri()) {
		callback('browser-mode');
		return () => {};
	}

	// Initial check
	getBackendStatus().then(callback);

	// Set up polling
	const intervalId = setInterval(async () => {
		const status = await getBackendStatus();
		callback(status);
	}, intervalMs);

	// Return cleanup function
	return () => clearInterval(intervalId);
}

/**
 * Open a native folder picker dialog
 * @param title - Dialog title
 * @param defaultPath - Default path to open
 * @returns Selected folder path or null if cancelled
 */
export async function selectFolder(title?: string, defaultPath?: string): Promise<string | null> {
	if (!isTauri()) {
		console.warn('Native folder picker not available in browser mode');
		return null;
	}

	try {
		const result = await open({
			directory: true,
			multiple: false,
			title: title,
			defaultPath: defaultPath
		});

		return result as string | null;
	} catch (error) {
		console.error('Failed to open folder picker:', error);
		return null;
	}
}

/**
 * Send a native OS notification
 * @param title - Notification title
 * @param body - Notification body text
 */
export async function showNotification(title: string, body: string): Promise<void> {
	if (!isTauri()) {
		// Fallback to browser notification
		if ('Notification' in window && Notification.permission === 'granted') {
			new Notification(title, { body });
		}
		return;
	}

	try {
		// Check and request permission if needed
		let permission = await isPermissionGranted();
		if (!permission) {
			const result = await requestPermission();
			permission = result === 'granted';
		}

		if (permission) {
			await sendNotification({ title, body });
		} else {
			console.warn('Notification permission not granted');
		}
	} catch (error) {
		console.error('Failed to send notification:', error);
	}
}
