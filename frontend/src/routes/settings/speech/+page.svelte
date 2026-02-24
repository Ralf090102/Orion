<script lang="ts">
	import { onMount } from "svelte";
	import CarbonTextToSpeech from "~icons/carbon/ibm-watson-text-to-speech";
	import CarbonMicrophone from "~icons/carbon/microphone";
	import CarbonSave from "~icons/carbon/save";
	import CarbonRenew from "~icons/carbon/renew";
	import CarbonCheckmark from "~icons/carbon/checkmark";
	import CarbonWarning from "~icons/carbon/warning";
	import CarbonPlay from "~icons/carbon/play";
	import CarbonDocument from "~icons/carbon/document";
	import CarbonStop from "~icons/carbon/stop";
	import CarbonTrash from "~icons/carbon/trash-can";
	import CarbonUpload from "~icons/carbon/upload";
	import CarbonRecordingFilled from "~icons/carbon/recording-filled";
	import CarbonSettings from "~icons/carbon/settings";
	import CarbonTextCreation from "~icons/carbon/text-creation";
	import CarbonUserSpeaker from "~icons/carbon/user-speaker";

	import { BACKEND_URL } from '$lib/utils/backendUrl';

	// Whisper STT configuration
	let whisperConfig = $state({
		model_size: 'base',
		device: 'auto',
		compute_type: 'int8',
		language: null as string | null,
		model_cache_dir: ''
	});

	// TTS configuration
	let ttsConfig = $state({
		default_voice: 'en_US-amy-low',
		audio_format: 'wav',
		default_speed: 1.0,
		use_gpu: false
	});

	let availableVoices = $state<any[]>([]);
	let savingTTS = $state(false);
	let loadingVoices = $state(false);
	let previewingVoice = $state(false);
	let currentPreviewAudio: HTMLAudioElement | null = null;

	// Engine management
	let currentEngine = $state<'piper' | 'qwen3'>('piper');
	let switchingEngine = $state(false);

	// Voice cloning state (Qwen3)
	let clonedVoices = $state<any[]>([]);
	let loadingClonedVoices = $state(false);
	let activeQwen3Voice = $state<string | null>(null);
	let cloningVoice = $state(false);
	let recording = $state(false);
	let recordingTime = $state(0);
	let mediaRecorder: MediaRecorder | null = null;
	let recordedChunks: Blob[] = [];
	let recordingInterval: number | null = null;
	
	// Voice cloning form
	let voiceCloneForm = $state({
		voice_name: '',
		ref_text: '',
		audio_file: null as File | null
	});

	// Synthesis test state (Qwen3)
	let synthesisTest = $state({
		text: '',
		voice_id: '',
		synthesizing: false,
		audioUrl: null as string | null
	});
	let currentSynthesisAudio: HTMLAudioElement | null = null;

	// Voice Design state (VoiceDesign model - generate voice from description)
	let voiceDesignForm = $state({
		voice_description: '',
		preview_text: 'Hello! How can I help you today?',
		save_voice_id: '',
		language: 'english'
	});
	let generatingVoice = $state(false);
	let savingDesignedVoice = $state(false);
	let designedAudioUrl = $state<string | null>(null);
	let currentDesignAudio: HTMLAudioElement | null = null;

	// Premium Speakers state (CustomVoice model - instruction-controlled synthesis)
	let customSpeakers = $state<Array<{
		speaker_id: string;
		name: string;
		description: string;
		gender: string;
		style: string;
	}>>([]);
	let loadingCustomSpeakers = $state(false);
	let customVoiceForm = $state({
		speaker: '',
		instruct: '',
		preview_text: 'Hello! How can I help you today?'
	});
	let previewingCustomVoice = $state(false);
	let customVoiceAudioUrl = $state<string | null>(null);
	let currentCustomAudio: HTMLAudioElement | null = null;

	// Qwen3 configuration
	let qwen3Config = $state({
		chunk_size: 200,
		auto_unload: true,
		unload_timeout_seconds: 300,
		model_loaded: false
	});
	let savingQwen3Config = $state(false);

	// UI state
	let loading = $state(false);
	let saving = $state(false);
	let testing = $state(false);
	let error = $state<string | null>(null);
	let success = $state<string | null>(null);
	let activeSection = $state('stt');
	let requiresReload = $state(false);
	let healthStatus = $state<any>(null);

	// Test audio state
	let testAudioFile = $state<File | null>(null);
	let testTranscription = $state<string>('');

	const sections = [
		{ id: 'stt', label: 'Speech-to-Text (STT)' },
		{ id: 'tts', label: 'Text-to-Speech (TTS)' },
		{ id: 'test', label: 'Test & Diagnostics' }
	];

	const modelSizes = [
		{ value: 'tiny', label: 'Tiny (fastest, least accurate)', size: '~75MB' },
		{ value: 'base', label: 'Base (balanced)', size: '~150MB' },
		{ value: 'small', label: 'Small (good quality)', size: '~500MB' },
		{ value: 'medium', label: 'Medium (high quality)', size: '~1.5GB' },
		{ value: 'large', label: 'Large (best quality)', size: '~3GB' },
		{ value: 'large-v2', label: 'Large v2 (improved)', size: '~3GB' },
		{ value: 'large-v3', label: 'Large v3 (latest)', size: '~3GB' }
	];

	const devices = [
		{ value: 'auto', label: 'Auto-detect' },
		{ value: 'cpu', label: 'CPU only' },
		{ value: 'cuda', label: 'CUDA (GPU)' }
	];

	const computeTypes = [
		{ value: 'int8', label: 'INT8 (fastest, less accurate)' },
		{ value: 'float16', label: 'Float16 (balanced, requires GPU)' },
		{ value: 'float32', label: 'Float32 (most accurate, slowest)' }
	];

	const commonLanguages = [
		{ value: null, label: 'Auto-detect' },
		{ value: 'en', label: 'English' },
		{ value: 'es', label: 'Spanish' },
		{ value: 'fr', label: 'French' },
		{ value: 'de', label: 'German' },
		{ value: 'it', label: 'Italian' },
		{ value: 'pt', label: 'Portuguese' },
		{ value: 'nl', label: 'Dutch' },
		{ value: 'ja', label: 'Japanese' },
		{ value: 'ko', label: 'Korean' },
		{ value: 'zh', label: 'Chinese' },
		{ value: 'ru', label: 'Russian' },
		{ value: 'ar', label: 'Arabic' }
	];

	async function loadWhisperConfig() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/whisper`);
			
			if (!response.ok) {
				throw new Error('Failed to load Whisper configuration');
			}

			const data = await response.json();
			whisperConfig = data.config;
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to load Whisper config:', err);
		} finally {
			loading = false;
		}
	}

	async function loadTTSConfig() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/tts`);
			
			if (!response.ok) {
				throw new Error('Failed to load TTS configuration');
			}

			const data = await response.json();
			ttsConfig = {
				default_voice: data.config.default_voice,
				audio_format: data.config.audio_format,
				default_speed: data.config.default_speed,
				use_gpu: data.config.use_gpu
			};
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to load TTS config:', err);
		} finally {
			loading = false;
		}
	}

	async function loadQwen3Config() {
		try {
			const response = await fetch(`${BACKEND_URL}/api/speech/config/qwen3`);
			
			if (!response.ok) {
				// Non-fatal: qwen3 may not be available
				console.warn('Could not load Qwen3 config (may not be enabled)');
				return;
			}

			const data = await response.json();
			qwen3Config = {
				chunk_size: data.config.chunk_size ?? 200,
				auto_unload: data.config.auto_unload ?? true,
				unload_timeout_seconds: data.config.unload_timeout_seconds ?? 300,
				model_loaded: data.config.model_loaded ?? false
			};
		} catch (err) {
			console.warn('Failed to load Qwen3 config:', err);
		}
	}

	async function saveQwen3Config() {
		try {
			savingQwen3Config = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/qwen3`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					chunk_size: qwen3Config.chunk_size,
					auto_unload: qwen3Config.auto_unload,
					unload_timeout_seconds: qwen3Config.unload_timeout_seconds
				})
			});

			if (!response.ok) {
				const errData = await response.json().catch(() => ({}));
				throw new Error(errData.detail || 'Failed to save Qwen3 configuration');
			}

			const data = await response.json();
			qwen3Config.model_loaded = data.config.model_loaded ?? qwen3Config.model_loaded;
			
			success = 'Qwen3 configuration saved';
			setTimeout(() => { success = null; }, 3000);
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to save Qwen3 config:', err);
		} finally {
			savingQwen3Config = false;
		}
	}

	async function loadAvailableVoices() {
		try {
			loadingVoices = true;
			console.log('Loading voices from:', `${BACKEND_URL}/api/speech/voices`);

			const response = await fetch(`${BACKEND_URL}/api/speech/voices`);
			
				if (!response.ok) {
					let serverMsg = '';
					try {
						const errJson = await response.json();
						serverMsg = errJson.detail || JSON.stringify(errJson);
					} catch (e) {
						serverMsg = await response.text();
					}
					console.error('Voice loading failed:', response.status, serverMsg);
					error = `Failed to load voices: ${serverMsg}`;
					return;
				}

				const data = await response.json();
				console.log('Voices loaded:', data);

				if (!data || !Array.isArray(data.voices)) {
					console.error('Unexpected voices payload', data);
					error = 'Unexpected voices payload from server';
					availableVoices = [];
					return;
				}

				// Normalize voice entries to a consistent shape so the UI can rely on fields
				availableVoices = data.voices.map((v: any) => ({
					voice_id: v.voice_id ?? v.id ?? v.name,
					name: v.name ?? v.voice_id ?? v.id ?? 'Unknown',
					language: v.language ?? v.lang ?? 'unknown',
					quality: v.quality ?? v.tier ?? 'medium',
					is_downloaded: v.is_downloaded ?? v.downloaded ?? v.local ?? false,
					model_size: v.model_size ?? v.size ?? '',
					gender: v.gender ?? '',
					description: v.description ?? v.note ?? ''
				}));
				console.log('Available voices count:', availableVoices.length);

			// If the configured default voice isn't present, pick the first available voice
			if (availableVoices.length > 0) {
				const found = availableVoices.find(v => v.voice_id === ttsConfig.default_voice);
				if (!found) {
					console.log('Default voice not found, using:', availableVoices[0].voice_id);
					ttsConfig = { ...ttsConfig, default_voice: availableVoices[0].voice_id };
				}
			}
		} catch (err) {
			console.error('Failed to load voices:', err);
			error = 'Failed to load TTS voices. Please check if backend is running.';
		} finally {
			loadingVoices = false;
		}
	}

	async function saveTTSConfig() {
		try {
			savingTTS = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/tts`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify(ttsConfig),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to save TTS configuration');
			}

			const data = await response.json();
			success = data.message;

			// Reload config to get updated values
			await loadTTSConfig();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to save TTS config:', err);
		} finally {
			savingTTS = false;
		}
	}

	async function changeVoice() {
		try {
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/voice`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					voice_id: ttsConfig.default_voice
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to change voice');
			}

			const data = await response.json();
			success = data.message;
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to change voice:', err);
		}
	}

	async function diagTestTTS() {
		if (currentEngine === 'qwen3') {
			await testSynthesisQwen3();
			return;
		}

		// Piper path: synthesize synthesisTest.text and show in audio player
		try {
			synthesisTest.synthesizing = true;
			synthesisTest.audioUrl = null;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/preview-voice`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					voice_id: ttsConfig.default_voice,
					text: synthesisTest.text.trim() || 'Hello, this is a text-to-speech diagnostic test.'
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Synthesis failed');
			}

			const audioBlob = await response.blob();
			synthesisTest.audioUrl = URL.createObjectURL(audioBlob);

			const audio = new Audio(synthesisTest.audioUrl);
			audio.play();
		} catch (err) {
			error = (err as Error).message;
		} finally {
			synthesisTest.synthesizing = false;
		}
	}

	async function previewVoice() {
		try {
			previewingVoice = true;
			error = null;

			// Stop current preview if playing
			if (currentPreviewAudio) {
				currentPreviewAudio.pause();
				currentPreviewAudio = null;
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/preview-voice`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					voice_id: ttsConfig.default_voice,
					text: 'Hello, this is a voice preview. How does this sound?'
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to preview voice');
			}

			const audioBlob = await response.blob();
			const audioUrl = URL.createObjectURL(audioBlob);
			
			currentPreviewAudio = new Audio(audioUrl);
			currentPreviewAudio.play();

			// Clean up URL when audio finishes
			currentPreviewAudio.onended = () => {
				URL.revokeObjectURL(audioUrl);
			};

		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to preview voice:', err);
		} finally {
			previewingVoice = false;
		}
	}

	async function saveWhisperConfig() {
		try {
			saving = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/config/whisper`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					model_size: whisperConfig.model_size,
					device: whisperConfig.device,
					compute_type: whisperConfig.compute_type,
					language: whisperConfig.language || null,
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to save configuration');
			}

			const data = await response.json();
			requiresReload = data.requires_reload;
			success = data.message;

			// Reload config to get updated values
			await loadWhisperConfig();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to save Whisper config:', err);
		} finally {
			saving = false;
		}
	}

	async function reloadWhisperModel() {
		// Model reloads automatically on next transcription
		// Just clear the reload flag
		requiresReload = false;
		success = 'Configuration saved. Model will reload automatically on next use.';
	}

	async function checkHealth() {
		try {
			loading = true;
			error = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/health`);
			
			if (!response.ok) {
				throw new Error('Failed to check speech health');
			}

			healthStatus = await response.json();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to check speech health:', err);
		} finally {
			loading = false;
		}
	}

	async function testTranscribe() {
		if (!testAudioFile) {
			error = 'Please select an audio file to test';
			return;
		}

		try {
			testing = true;
			error = null;
			testTranscription = '';

			const formData = new FormData();
			formData.append('audio', testAudioFile);
			if (whisperConfig.language) {
				formData.append('language', whisperConfig.language);
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/transcribe`, {
				method: 'POST',
				body: formData,
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Transcription failed');
			}

			const data = await response.json();
			testTranscription = data.text;
			success = `Transcribed ${data.duration.toFixed(1)}s of ${data.language} audio`;
		} catch (err) {
			error = (err as Error).message;
			console.error('Transcription test failed:', err);
		} finally {
			testing = false;
		}
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			testAudioFile = target.files[0];
			testTranscription = '';
			error = null;
		}
	}

	// ========== ENGINE SWITCHING ==========
	async function switchEngine(newEngine: 'piper' | 'qwen3') {
		try {
			switchingEngine = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/engine`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ engine: newEngine }),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to switch engine');
			}

			const data = await response.json();
			currentEngine = newEngine;
			success = data.message;

			// Reload appropriate voice lists
			if (newEngine === 'piper') {
				await loadAvailableVoices();
			} else {
				await loadClonedVoices();
			}

			// Refresh health panel to reflect new active engine
			await checkHealth();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to switch engine:', err);
		} finally {
			switchingEngine = false;
		}
	}

	// ========== VOICE CLONING ==========
	async function loadClonedVoices() {
		try {
			loadingClonedVoices = true;
			const response = await fetch(`${BACKEND_URL}/api/speech/cloned-voices`);
			
			if (!response.ok) {
				const errorData = await response.json();
				// If engine is piper, this is expected (Qwen3 endpoint blocked)
				if (currentEngine === 'piper') {
					console.log('Cloned voices not available (Piper engine active)');
					clonedVoices = [];
					return;
				}
				throw new Error(errorData.detail || 'Failed to load cloned voices');
			}

			const data = await response.json();
			clonedVoices = data.voices || [];

			// Auto-select first voice in synthesis test if none selected
			if (clonedVoices.length > 0 && !synthesisTest.voice_id) {
				synthesisTest.voice_id = clonedVoices[0].voice_id;
			}

			// Also refresh the active voice indicator
			await loadActiveVoice();
		} catch (err) {
			if (currentEngine !== 'piper') {
				console.error('Failed to load cloned voices:', err);
				error = (err as Error).message;
			}
		} finally {
			loadingClonedVoices = false;
		}
	}

	async function loadActiveVoice() {
		try {
			const response = await fetch(`${BACKEND_URL}/api/speech/active-voice`);
			if (response.ok) {
				const data = await response.json();
				activeQwen3Voice = data.active_voice ?? null;
			}
		} catch (err) {
			console.error('Failed to load active voice:', err);
		}
	}

	async function setActiveVoice(voiceId: string) {
		try {
			error = null;
			const response = await fetch(`${BACKEND_URL}/api/speech/active-voice`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ voice_id: voiceId }),
			});
			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to set active voice');
			}
			const data = await response.json();
			activeQwen3Voice = data.active_voice ?? null;
			success = data.message;
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to set active voice:', err);
		}
	}

	async function startRecording() {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			mediaRecorder = new MediaRecorder(stream);
			recordedChunks = [];
			recordingTime = 0;

			mediaRecorder.ondataavailable = (event) => {
				if (event.data.size > 0) {
					recordedChunks.push(event.data);
				}
			};

			mediaRecorder.onstop = () => {
				const blob = new Blob(recordedChunks, { type: 'audio/webm' });
				const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
				voiceCloneForm.audio_file = file;
				
				// Stop all tracks
				stream.getTracks().forEach(track => track.stop());
			};

			mediaRecorder.start();
			recording = true;

			// Start timer
			recordingInterval = window.setInterval(() => {
				recordingTime++;
				// Auto-stop at 15 seconds
				if (recordingTime >= 15) {
					stopRecording();
				}
			}, 1000);
		} catch (err) {
			error = 'Failed to access microphone. Please check permissions.';
			console.error('Recording error:', err);
		}
	}

	function stopRecording() {
		if (mediaRecorder && recording) {
			mediaRecorder.stop();
			recording = false;
			if (recordingInterval) {
				clearInterval(recordingInterval);
				recordingInterval = null;
			}
		}
	}

	function handleVoiceFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			voiceCloneForm.audio_file = target.files[0];
			error = null;
		}
	}

	async function cloneVoice() {
		if (!voiceCloneForm.voice_name || !voiceCloneForm.audio_file) {
			error = 'Please provide a voice name and audio file';
			return;
		}

		try {
			cloningVoice = true;
			error = null;
			success = null;

			const formData = new FormData();
			formData.append('voice_name', voiceCloneForm.voice_name);
			if (voiceCloneForm.ref_text) {
				formData.append('ref_text', voiceCloneForm.ref_text);
			}
			formData.append('audio', voiceCloneForm.audio_file);

			const response = await fetch(`${BACKEND_URL}/api/speech/clone-voice`, {
				method: 'POST',
				body: formData,
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Voice cloning failed');
			}

			const data = await response.json();
			success = data.message;

			// Reset form
			voiceCloneForm = {
				voice_name: '',
				ref_text: '',
				audio_file: null
			};
			recordingTime = 0;

			// Reload cloned voices
			await loadClonedVoices();
		} catch (err) {
			error = (err as Error).message;
			console.error('Voice cloning failed:', err);
		} finally {
			cloningVoice = false;
		}
	}

	// ========== VOICE DESIGN (VoiceDesign model) ==========
	async function generateVoicePreview() {
		if (!voiceDesignForm.voice_description.trim()) {
			error = 'Please provide a voice description';
			return;
		}
		if (!voiceDesignForm.preview_text.trim()) {
			error = 'Please provide text to preview';
			return;
		}

		try {
			generatingVoice = true;
			error = null;
			success = null;

			// Stop any currently playing audio
			if (currentDesignAudio) {
				currentDesignAudio.pause();
				currentDesignAudio = null;
			}
			if (designedAudioUrl) {
				URL.revokeObjectURL(designedAudioUrl);
				designedAudioUrl = null;
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/generate-voice`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text: voiceDesignForm.preview_text,
					voice_description: voiceDesignForm.voice_description,
					language: voiceDesignForm.language
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Voice generation failed');
			}

			const blob = await response.blob();
			designedAudioUrl = URL.createObjectURL(blob);

			// Auto-play
			currentDesignAudio = new Audio(designedAudioUrl);
			currentDesignAudio.play();
			success = 'Voice preview generated!';
		} catch (err) {
			error = (err as Error).message;
			console.error('Voice generation failed:', err);
		} finally {
			generatingVoice = false;
		}
	}

	async function designAndSaveVoice() {
		if (!voiceDesignForm.voice_description.trim()) {
			error = 'Please provide a voice description';
			return;
		}
		if (!voiceDesignForm.save_voice_id.trim()) {
			error = 'Please provide a voice ID to save';
			return;
		}
		if (!voiceDesignForm.preview_text.trim()) {
			error = 'Please provide sample text for the voice';
			return;
		}

		try {
			savingDesignedVoice = true;
			error = null;
			success = null;

			const response = await fetch(`${BACKEND_URL}/api/speech/design-and-save`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					voice_description: voiceDesignForm.voice_description,
					voice_id: voiceDesignForm.save_voice_id,
					sample_text: voiceDesignForm.preview_text,
					language: voiceDesignForm.language
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to save designed voice');
			}

			const data = await response.json();
			success = data.message;

			// Reset save voice name
			voiceDesignForm.save_voice_id = '';

			// Reload voices to show the new one
			await loadClonedVoices();
		} catch (err) {
			error = (err as Error).message;
			console.error('Design and save failed:', err);
		} finally {
			savingDesignedVoice = false;
		}
	}

	// ========== PREMIUM SPEAKERS (CustomVoice model) ==========
	async function loadCustomSpeakers() {
		try {
			loadingCustomSpeakers = true;
			const response = await fetch(`${BACKEND_URL}/api/speech/custom-speakers`);
			
			if (!response.ok) {
				throw new Error('Failed to load custom speakers');
			}

			const data = await response.json();
			customSpeakers = data.speakers || [];
		} catch (err) {
			console.error('Failed to load custom speakers:', err);
		} finally {
			loadingCustomSpeakers = false;
		}
	}

	async function previewCustomVoice() {
		if (!customVoiceForm.speaker) {
			error = 'Please select a speaker';
			return;
		}
		if (!customVoiceForm.preview_text.trim()) {
			error = 'Please provide text to preview';
			return;
		}

		try {
			previewingCustomVoice = true;
			error = null;
			success = null;

			// Stop any currently playing audio
			if (currentCustomAudio) {
				currentCustomAudio.pause();
				currentCustomAudio = null;
			}
			if (customVoiceAudioUrl) {
				URL.revokeObjectURL(customVoiceAudioUrl);
				customVoiceAudioUrl = null;
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/custom-voice`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text: customVoiceForm.preview_text,
					speaker: customVoiceForm.speaker,
					instruct: customVoiceForm.instruct || undefined
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Custom voice synthesis failed');
			}

			const blob = await response.blob();
			customVoiceAudioUrl = URL.createObjectURL(blob);

			// Auto-play
			currentCustomAudio = new Audio(customVoiceAudioUrl);
			currentCustomAudio.play();
			
			const speakerInfo = customSpeakers.find(s => s.speaker_id === customVoiceForm.speaker);
			success = `Preview generated with ${speakerInfo?.name || customVoiceForm.speaker}`;
		} catch (err) {
			error = (err as Error).message;
			console.error('Custom voice preview failed:', err);
		} finally {
			previewingCustomVoice = false;
		}
	}

	async function deleteClonedVoice(voiceId: string) {
		if (!confirm(`Delete voice "${voiceId}"? This cannot be undone.`)) {
			return;
		}

		try {
			const response = await fetch(`${BACKEND_URL}/api/speech/cloned-voices/${voiceId}`, {
				method: 'DELETE',
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to delete voice');
			}

			success = `Voice "${voiceId}" deleted successfully`;
			await loadClonedVoices();
		} catch (err) {
			error = (err as Error).message;
			console.error('Failed to delete voice:', err);
		}
	}

	async function testSynthesisQwen3() {
		if (!synthesisTest.text.trim()) {
			error = 'Please enter text to synthesize';
			return;
		}
		if (!synthesisTest.voice_id) {
			error = 'Please select a voice to test';
			return;
		}

		try {
			synthesisTest.synthesizing = true;
			error = null;

			// Stop any currently playing audio
			if (currentSynthesisAudio) {
				currentSynthesisAudio.pause();
				currentSynthesisAudio = null;
			}
			if (synthesisTest.audioUrl) {
				URL.revokeObjectURL(synthesisTest.audioUrl);
				synthesisTest.audioUrl = null;
			}

			const response = await fetch(`${BACKEND_URL}/api/speech/synthesize-qwen3`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text: synthesisTest.text,
					voice_id: synthesisTest.voice_id,
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Synthesis failed');
			}

			const blob = await response.blob();
			synthesisTest.audioUrl = URL.createObjectURL(blob);

			// Auto-play
			currentSynthesisAudio = new Audio(synthesisTest.audioUrl);
			currentSynthesisAudio.play();
			success = `Synthesized with voice "${synthesisTest.voice_id}"`;
		} catch (err) {
			error = (err as Error).message;
			console.error('Synthesis test failed:', err);
		} finally {
			synthesisTest.synthesizing = false;
		}
	}

	async function getEngineStatus() {
		try {
			const response = await fetch(`${BACKEND_URL}/api/status`);
			if (response.ok) {
				const data = await response.json();
				currentEngine = data.tts_engine || 'piper';
			}
		} catch (err) {
			console.error('Failed to get engine status:', err);
		}
	}

	onMount(async () => {
		await getEngineStatus();
		await loadWhisperConfig();
		await loadTTSConfig();
		await loadQwen3Config();
		await checkHealth();
		await loadActiveVoice();
		
		// Load voices based on current engine
		if (currentEngine === 'piper') {
			await loadAvailableVoices();
		} else {
			await loadClonedVoices();
			await loadCustomSpeakers();
		}
	});
</script>

<svelte:head>
	<title>Speech & Audio Settings - Orion</title>
</svelte:head>

<div class="flex h-full flex-col gap-y-6 overflow-y-auto px-5 py-8 sm:px-8">
	<!-- Header -->
	<div>
		<h1 class="text-2xl font-bold">Speech & Audio Settings</h1>
		<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
			Configure speech-to-text (STT) and text-to-speech (TTS) capabilities
		</p>
	</div>

	<!-- Tabs -->
	<div class="border-b border-gray-200 dark:border-gray-700">
		<nav class="-mb-px flex space-x-8">
			{#each sections as section}
				<button
					onclick={() => { activeSection = section.id; error = null; success = null; }}
					class="whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors
						{activeSection === section.id
							? 'border-blue-500 text-blue-600 dark:border-blue-400 dark:text-blue-400'
							: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:border-gray-600 dark:hover:text-gray-300'}"
				>
					{section.label}
				</button>
			{/each}
		</nav>
	</div>

	<!-- Alerts -->
	{#if error}
		<div class="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
			<div class="flex items-start gap-3">
				<CarbonWarning class="size-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<p class="text-sm font-medium text-red-800 dark:text-red-200">Error</p>
					<p class="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
				</div>
			</div>
		</div>
	{/if}

	{#if success}
		<div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
			<div class="flex items-start gap-3">
				<CarbonCheckmark class="size-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<p class="text-sm font-medium text-green-800 dark:text-green-200">Success</p>
					<p class="text-sm text-green-700 dark:text-green-300 mt-1">{success}</p>
				</div>
			</div>
		</div>
	{/if}

	{#if requiresReload}
		<div class="rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
			<div class="flex items-start justify-between gap-3">
				<div class="flex items-start gap-3">
					<CarbonWarning class="size-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
					<div class="flex-1">
						<p class="text-sm font-medium text-yellow-800 dark:text-yellow-200">Model Reload Required</p>
						<p class="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
							Configuration changes require reloading the Whisper model to take effect.
						</p>
					</div>
				</div>
				<button
					onclick={reloadWhisperModel}
					disabled={loading}
					class="rounded-lg bg-yellow-600 px-4 py-2 text-sm font-medium text-white hover:bg-yellow-700 disabled:opacity-50 dark:bg-yellow-500 dark:hover:bg-yellow-600"
				>
					<div class="flex items-center gap-2">
						<CarbonRenew class="size-4" />
						Reload Model
					</div>
				</button>
			</div>
		</div>
	{/if}

	<!-- Content Sections -->
	{#if activeSection === 'stt'}
		<!-- Speech-to-Text Configuration -->
		<div class="space-y-6">
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-blue-100 p-3 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
						<CarbonMicrophone class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Whisper STT Configuration</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Configure the Whisper speech-to-text model
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Model Size -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Model Size
						</label>
						<select
							bind:value={whisperConfig.model_size}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each modelSizes as size}
								<option value={size.value}>
									{size.label} - {size.size}
								</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Larger models provide better accuracy but require more memory and are slower.
						</p>
					</div>

					<!-- Device -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Compute Device
						</label>
						<select
							bind:value={whisperConfig.device}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each devices as device}
								<option value={device.value}>{device.label}</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Auto-detect will use GPU if available, otherwise CPU.
						</p>
					</div>

					<!-- Compute Type -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Compute Precision
						</label>
						<select
							bind:value={whisperConfig.compute_type}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each computeTypes as type}
								<option value={type.value}>{type.label}</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Float16 requires GPU. INT8 is recommended for CPU.
						</p>
					</div>

					<!-- Language -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Default Language
						</label>
						<select
							bind:value={whisperConfig.language}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							{#each commonLanguages as lang}
								<option value={lang.value}>
									{lang.label}
								</option>
							{/each}
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Set to auto-detect for multi-language support, or specify a language for better accuracy.
						</p>
					</div>

					<!-- Model Cache Directory -->
					{#if whisperConfig.model_cache_dir}
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
								Model Cache Directory
							</label>
							<div class="rounded-lg border border-gray-300 bg-gray-50 px-4 py-2.5 font-mono text-sm text-gray-700 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-300">
								{whisperConfig.model_cache_dir}
							</div>
							<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
								Downloaded Whisper models are cached here.
							</p>
						</div>
					{/if}
				</div>

				<!-- Save Button -->
				<div class="mt-8 flex justify-end">
					<button
						onclick={saveWhisperConfig}
						disabled={saving || loading}
						class="rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
					>
						<div class="flex items-center gap-2">
							{#if saving}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonSave class="size-4" />
							{/if}
							{saving ? 'Saving...' : 'Save Configuration'}
						</div>
					</button>
				</div>
			</div>

			<!-- Test Transcription -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="mb-6">
					<h2 class="text-lg font-semibold">Test Transcription</h2>
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Upload an audio file to test the Whisper STT system
					</p>
				</div>

				<div class="space-y-4">
					<!-- File Upload -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio File
						</label>
						<input
							type="file"
							accept="audio/*,.webm,.wav,.mp3,.m4a,.ogg,.flac"
							onchange={handleFileSelect}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 file:mr-4 file:rounded file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 dark:file:bg-blue-900/30 dark:file:text-blue-400"
						/>
						{#if testAudioFile}
							<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
								Selected: {testAudioFile.name} ({(testAudioFile.size / 1024).toFixed(1)} KB)
							</p>
						{/if}
					</div>

					<!-- Test Button -->
					<button
						onclick={testTranscribe}
						disabled={!testAudioFile || testing}
						class="w-full rounded-lg bg-green-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50 dark:bg-green-500 dark:hover:bg-green-600"
					>
						<div class="flex items-center justify-center gap-2">
							{#if testing}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonPlay class="size-4" />
							{/if}
							{testing ? 'Transcribing...' : 'Test Transcription'}
						</div>
					</button>

					<!-- Transcription Result -->
					{#if testTranscription}
						<div class="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
							<p class="text-sm font-medium text-green-800 dark:text-green-200 mb-2">
								Transcription Result:
							</p>
							<p class="text-sm text-green-700 dark:text-green-300 whitespace-pre-wrap">
								{testTranscription}
							</p>
						</div>
					{/if}
				</div>
			</div>
		</div>


		<!-- Info Box -->
		<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
			<div class="flex items-start gap-3">
				<CarbonDocument class="size-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
				<div class="flex-1">
					<h3 class="font-semibold text-blue-900 dark:text-blue-200 text-sm">
						About Whisper STT
					</h3>
					<p class="text-sm text-blue-800 dark:text-blue-300 mt-1">
						Whisper is OpenAI's automatic speech recognition system. The first time you use a model size,
						it will be downloaded automatically. Larger models provide better accuracy but require more
						resources. The 'base' model is a good balance for most users.
					</p>
				</div>
			</div>
		</div>

	{:else if activeSection === 'tts'}
		<!-- Text-to-Speech Configuration -->
		<div class="space-y-6">
			<!-- Engine Selector -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="mb-4">
					<h2 class="text-lg font-semibold mb-2">TTS Engine</h2>
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Choose between Piper (fast, pre-built voices) and Qwen3 (voice cloning)
					</p>
				</div>

				<div class="flex gap-4">
					<button
						onclick={() => switchEngine('piper')}
						disabled={switchingEngine || currentEngine === 'piper'}
						class="flex-1 flex items-center justify-center gap-3 rounded-lg border-2 px-6 py-4 transition-all
							{currentEngine === 'piper'
								? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
								: 'border-gray-300 dark:border-gray-600 hover:border-blue-400'}
							disabled:opacity-50"
					>
						<CarbonPlay class="size-5" />
						<div class="text-left">
							<div class="font-semibold">Piper TTS</div>
							<div class="text-xs text-gray-600 dark:text-gray-400">Fast, CPU-friendly</div>
						</div>
						{#if currentEngine === 'piper'}
							<CarbonCheckmark class="size-5 text-blue-600 dark:text-blue-400 ml-auto" />
						{/if}
					</button>

					<button
						onclick={() => switchEngine('qwen3')}
						disabled={switchingEngine || currentEngine === 'qwen3'}
						class="flex-1 flex items-center justify-center gap-3 rounded-lg border-2 px-6 py-4 transition-all
							{currentEngine === 'qwen3'
								? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
								: 'border-gray-300 dark:border-gray-600 hover:border-purple-400'}
							disabled:opacity-50"
					>
						<CarbonMicrophone class="size-5" />
						<div class="text-left">
							<div class="font-semibold">Qwen3-TTS</div>
							<div class="text-xs text-gray-600 dark:text-gray-400">Voice cloning, GPU required</div>
						</div>
						{#if currentEngine === 'qwen3'}
							<CarbonCheckmark class="size-5 text-purple-600 dark:text-purple-400 ml-auto" />
						{/if}
					</button>
				</div>

				{#if switchingEngine}
					<div class="mt-4 flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-400">
						<CarbonRenew class="size-4 animate-spin" />
						<span>Switching engine...</span>
					</div>
				{/if}
			</div>

			<!-- Piper TTS Configuration -->
			{#if currentEngine === 'piper'}
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-purple-100 p-3 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
						<CarbonTextToSpeech class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Piper TTS Configuration</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Configure Piper TTS voices and audio settings
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Voice Selection -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Default Voice
						</label>
						<div class="flex gap-3">
							<select
								bind:value={ttsConfig.default_voice}
								onchange={changeVoice}
								disabled={loadingVoices}
								class="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							>
								{#if loadingVoices}
									<option>Loading voices...</option>
								{:else if availableVoices.length === 0}
									<option>No voices available</option>
								{:else}
									{#each availableVoices as voice}
										<option value={voice.voice_id}>
											{voice.name} ({voice.language}) - {voice.quality}
											{#if voice.is_downloaded}✓{/if}
										</option>
									{/each}
								{/if}
							</select>
							<button
								onclick={previewVoice}
								disabled={previewingVoice || loadingVoices || availableVoices.length === 0}
								class="rounded-lg bg-purple-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600 whitespace-nowrap"
							>
								<div class="flex items-center gap-2">
									{#if previewingVoice}
										<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
									{:else}
										<CarbonPlay class="size-4" />
									{/if}
									Preview
								</div>
							</button>
						</div>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Select a voice and click Preview to hear a sample. Voices marked with ✓ are downloaded locally.
						</p>
					</div>

					<!-- Voice Info -->
					{#if !loadingVoices && availableVoices.length > 0}
						{@const selectedVoice = availableVoices.find(v => v.voice_id === ttsConfig.default_voice)}
						{#if selectedVoice}
							<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50">
								<div class="grid grid-cols-2 gap-4 text-sm">
									<div>
										<span class="text-gray-600 dark:text-gray-400">Gender:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100 capitalize">{selectedVoice.gender}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Quality:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100 capitalize">{selectedVoice.quality}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Language:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100">{selectedVoice.language}</span>
									</div>
									<div>
										<span class="text-gray-600 dark:text-gray-400">Model Size:</span>
										<span class="ml-2 font-medium text-gray-900 dark:text-gray-100">{selectedVoice.model_size}</span>
									</div>
								</div>
								<p class="text-xs text-gray-600 dark:text-gray-400 mt-3">
									{selectedVoice.description}
								</p>
							</div>
						{/if}
					{/if}

					<!-- Speed Control -->
					<div>
						<div class="flex justify-between items-center mb-2">
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
								Speech Speed
							</label>
							<span class="text-sm font-mono text-gray-600 dark:text-gray-400">
								{ttsConfig.default_speed.toFixed(1)}x
							</span>
						</div>
						<input 
							type="range" 
							min="0.0" 
							max="2.0" 
							step="0.1"
							bind:value={ttsConfig.default_speed}
							class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-purple-600"
						/>
						<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
							<span>0.0x (Slowest)</span>
							<span>1.0x (Normal)</span>
							<span>2.0x (Fastest)</span>
						</div>
					</div>

					<!-- Audio Format -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio Format
						</label>
						<select
							bind:value={ttsConfig.audio_format}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="wav">WAV (Uncompressed, Higher Quality)</option>
							<option value="mp3">MP3 (Compressed, Smaller Size)</option>
						</select>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							WAV provides best quality but larger file sizes. MP3 is compressed and more efficient.
						</p>
					</div>

					<!-- GPU Acceleration -->
					<div class="rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50">
						<label class="flex items-start gap-3 cursor-pointer">
							<input 
								type="checkbox" 
								bind:checked={ttsConfig.use_gpu}
								class="mt-0.5 size-4 rounded border-gray-300 text-purple-600 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700"
							/>
							<div class="flex-1">
								<div class="font-medium text-gray-900 dark:text-gray-100">
									Enable GPU Acceleration (CUDA)
								</div>
								<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
									Use GPU for faster TTS synthesis. Requires NVIDIA GPU with CUDA support.
									Disable if you encounter errors or don't have a compatible GPU.
								</p>
							</div>
						</label>
					</div>
				</div>

				<!-- Save Button -->
				<div class="mt-8 flex justify-end">
					<button
						onclick={saveTTSConfig}
						disabled={savingTTS || loading}
						class="rounded-lg bg-purple-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600"
					>
						<div class="flex items-center gap-2">
							{#if savingTTS}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							{:else}
								<CarbonSave class="size-4" />
							{/if}
							{savingTTS ? 'Saving...' : 'Save Configuration'}
						</div>
					</button>
				</div>
			</div>

			<!-- Info Box -->
			<div class="rounded-lg border border-purple-200 bg-purple-50 p-4 dark:border-purple-800 dark:bg-purple-900/20">
				<div class="flex items-start gap-3">
					<CarbonDocument class="size-5 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
					<div class="flex-1">
						<h3 class="font-semibold text-purple-900 dark:text-purple-200 text-sm">
							About Piper TTS
						</h3>
						<p class="text-sm text-purple-800 dark:text-purple-300 mt-1">
							Piper is a fast, local neural text-to-speech system. Voice models are automatically loaded on first use.
							The system supports multiple languages and voice qualities. For best performance on CPU, use low or medium quality voices.
						</p>
					</div>
				</div>
			</div>
			{/if}

			<!-- Qwen3-TTS Voice Cloning -->
			{#if currentEngine === 'qwen3'}
			<!-- Voice Cloning Interface -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-purple-100 p-3 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
						<CarbonMicrophone class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Voice Cloning</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Clone a voice from 3-15 second audio sample
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Voice Name -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Voice Name <span class="text-red-500">*</span>
						</label>
						<input
							type="text"
							bind:value={voiceCloneForm.voice_name}
							placeholder="e.g., my_voice, john_doe"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Unique identifier for this voice (letters, numbers, underscores only)
						</p>
					</div>

					<!-- Reference Text (Optional) -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Reference Text (Recommended)
						</label>
						<textarea
							bind:value={voiceCloneForm.ref_text}
							placeholder="The exact text spoken in the audio sample..."
							rows="3"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						></textarea>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Providing the transcript improves voice cloning quality (ICL mode)
						</p>
					</div>

					<!-- Audio Input -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio Sample <span class="text-red-500">*</span>
						</label>

						<div class="space-y-3">
							<!-- Record Button -->
							<div>
								<button
									onclick={recording ? stopRecording : startRecording}
									disabled={cloningVoice}
									class="w-full flex items-center justify-center gap-3 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 px-6 py-4 hover:border-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/10 transition-colors disabled:opacity-50"
								>
									{#if recording}
										<CarbonStop class="size-5 text-red-600" />
										<div class="text-left">
											<div class="font-semibold text-red-600">Recording... {recordingTime}s / 15s</div>
											<div class="text-xs text-gray-600 dark:text-gray-400">Click to stop</div>
										</div>
									{:else}
										<CarbonRecordingFilled class="size-5 text-purple-600" />
										<div class="text-left">
											<div class="font-semibold">Record Audio (3-15s)</div>
											<div class="text-xs text-gray-600 dark:text-gray-400">Click to start recording</div>
										</div>
									{/if}
								</button>
								{#if recording}
									<div class="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
										<div 
											class="h-full bg-red-600 transition-all duration-1000"
											style="width: {(recordingTime / 15) * 100}%"
										></div>
									</div>
								{/if}
							</div>

							<!-- Or Divider -->
							<div class="flex items-center gap-3">
								<div class="flex-1 h-px bg-gray-300 dark:bg-gray-600"></div>
								<span class="text-sm text-gray-500 dark:text-gray-400">or</span>
								<div class="flex-1 h-px bg-gray-300 dark:bg-gray-600"></div>
							</div>

							<!-- Upload Button -->
							<label class="block">
								<input
									type="file"
									accept="audio/*"
									onchange={handleVoiceFileSelect}
									disabled={recording || cloningVoice}
									class="hidden"
								/>
								<div class="w-full flex items-center justify-center gap-3 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 px-6 py-4 hover:border-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/10 transition-colors cursor-pointer">
									<CarbonUpload class="size-5 text-purple-600" />
									<div class="text-left">
										<div class="font-semibold">Upload Audio File</div>
										<div class="text-xs text-gray-600 dark:text-gray-400">
											{voiceCloneForm.audio_file ? voiceCloneForm.audio_file.name : 'WAV, MP3, or WebM (3-15s)'}
										</div>
									</div>
								</div>
							</label>
						</div>
					</div>

					<!-- Clone Button -->
					<div class="flex justify-end">
						<button
							onclick={cloneVoice}
							disabled={cloningVoice || !voiceCloneForm.voice_name || !voiceCloneForm.audio_file}
							class="rounded-lg bg-purple-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600"
						>
							<div class="flex items-center gap-2">
								{#if cloningVoice}
									<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
								{:else}
									<CarbonMicrophone class="size-4" />
								{/if}
								{cloningVoice ? 'Cloning Voice...' : 'Clone Voice'}
							</div>
						</button>
					</div>
				</div>
			</div>

			<!-- Voice Design Interface (VoiceDesign Model) -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-indigo-100 p-3 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400">
						<CarbonTextCreation class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Design Voice from Description</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Generate a voice from text description (no audio sample needed)
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Voice Description -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Voice Description <span class="text-red-500">*</span>
						</label>
						<textarea
							bind:value={voiceDesignForm.voice_description}
							placeholder="Describe the voice you want to create...
e.g., 'A warm, friendly female voice with a slight British accent, suitable for audiobooks.'
or 'A deep, authoritative male voice with a calm and reassuring tone.'"
							rows="4"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						></textarea>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Be specific about gender, age, accent, tone, and speaking style for best results
						</p>
					</div>

					<!-- Preview Text -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Preview Text
						</label>
						<textarea
							bind:value={voiceDesignForm.preview_text}
							placeholder="Enter text to hear the designed voice..."
							rows="2"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						></textarea>
					</div>

					<!-- Language Selection -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Language
						</label>
						<select
							bind:value={voiceDesignForm.language}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="english">English</option>
							<option value="chinese">Chinese</option>
							<option value="japanese">Japanese</option>
							<option value="korean">Korean</option>
							<option value="french">French</option>
							<option value="german">German</option>
							<option value="spanish">Spanish</option>
						</select>
					</div>

					<!-- Audio Preview -->
					{#if designedAudioUrl}
						<div class="rounded-lg border border-indigo-200 bg-indigo-50 dark:border-indigo-800 dark:bg-indigo-900/20 p-4">
							<div class="flex items-center gap-3">
								<audio
									src={designedAudioUrl}
									controls
									class="flex-1 h-10"
								></audio>
							</div>
						</div>
					{/if}

					<!-- Actions Row -->
					<div class="flex flex-wrap items-center justify-between gap-3">
						<button
							onclick={generateVoicePreview}
							disabled={generatingVoice || !voiceDesignForm.voice_description.trim() || !voiceDesignForm.preview_text.trim()}
							class="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
						>
							<div class="flex items-center gap-2">
								{#if generatingVoice}
									<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
									Generating...
								{:else}
									<CarbonPlay class="size-4" />
									Preview Voice
								{/if}
							</div>
						</button>

						<!-- Save Voice Section -->
						{#if designedAudioUrl}
							<div class="flex items-center gap-2 flex-1 justify-end">
								<input
									type="text"
									bind:value={voiceDesignForm.save_voice_id}
									placeholder="Voice ID to save..."
									class="w-48 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
								/>
								<button
									onclick={designAndSaveVoice}
									disabled={savingDesignedVoice || !voiceDesignForm.save_voice_id.trim()}
									class="rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50 dark:bg-green-500 dark:hover:bg-green-600"
								>
									<div class="flex items-center gap-2">
										{#if savingDesignedVoice}
											<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
											Saving...
										{:else}
											<CarbonSave class="size-4" />
											Save to Library
										{/if}
									</div>
								</button>
							</div>
						{/if}
					</div>
				</div>
			</div>

			<!-- Premium Speakers Interface (CustomVoice Model) -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-amber-100 p-3 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400">
						<CarbonUserSpeaker class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Premium Speakers</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							High-quality pre-trained voices with instruction control
						</p>
					</div>
				</div>

				<div class="space-y-6">
					<!-- Speaker Selection -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Select Speaker <span class="text-red-500">*</span>
						</label>
						{#if loadingCustomSpeakers}
							<div class="flex items-center gap-2 text-gray-500 dark:text-gray-400 py-2">
								<CarbonRenew class="size-4 animate-spin" />
								Loading speakers...
							</div>
						{:else if customSpeakers.length === 0}
							<div class="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20 p-4">
								<p class="text-sm text-amber-800 dark:text-amber-200">
									Premium speakers not available. Make sure the CustomVoice model is configured.
								</p>
								<button
									onclick={loadCustomSpeakers}
									class="mt-2 text-sm text-amber-600 hover:text-amber-700 dark:text-amber-400"
								>
									Retry loading speakers
								</button>
							</div>
						{:else}
							<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
								{#each customSpeakers as speaker}
									<button
										onclick={() => { customVoiceForm.speaker = speaker.speaker_id; }}
										class="text-left p-3 rounded-lg border-2 transition-all
											{customVoiceForm.speaker === speaker.speaker_id
												? 'border-amber-400 bg-amber-50 dark:border-amber-500 dark:bg-amber-900/20'
												: 'border-gray-200 dark:border-gray-700 hover:border-amber-300 dark:hover:border-amber-600'}"
									>
										<div class="font-medium text-gray-900 dark:text-gray-100">{speaker.name}</div>
										<div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
											{speaker.gender} • {speaker.style}
										</div>
										<div class="text-xs text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
											{speaker.description}
										</div>
									</button>
								{/each}
							</div>
						{/if}
					</div>

					<!-- Instruction (Optional) -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Style Instruction (Optional)
						</label>
						<input
							type="text"
							bind:value={customVoiceForm.instruct}
							placeholder="e.g., 'Speak with excitement' or 'Read slowly and calmly'"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-amber-500 focus:ring-2 focus:ring-amber-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						/>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Control how the speaker delivers the text (emotion, pace, emphasis)
						</p>
					</div>

					<!-- Preview Text -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Text to Speak <span class="text-red-500">*</span>
						</label>
						<textarea
							bind:value={customVoiceForm.preview_text}
							placeholder="Enter text to hear the premium speaker..."
							rows="2"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-amber-500 focus:ring-2 focus:ring-amber-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						></textarea>
					</div>

					<!-- Audio Preview -->
					{#if customVoiceAudioUrl}
						<div class="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20 p-4">
							<audio
								src={customVoiceAudioUrl}
								controls
								class="w-full h-10"
							></audio>
						</div>
					{/if}

					<!-- Preview Button -->
					<div class="flex justify-end">
						<button
							onclick={previewCustomVoice}
							disabled={previewingCustomVoice || !customVoiceForm.speaker || !customVoiceForm.preview_text.trim()}
							class="rounded-lg bg-amber-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50 dark:bg-amber-500 dark:hover:bg-amber-600"
						>
							<div class="flex items-center gap-2">
								{#if previewingCustomVoice}
									<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
									Generating...
								{:else}
									<CarbonPlay class="size-4" />
									Preview Speaker
								{/if}
							</div>
						</button>
					</div>
				</div>
			</div>

			<!-- Cloned Voices List -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center justify-between mb-4">
					<h2 class="text-lg font-semibold">Voice Library</h2>
					<button
						onclick={loadClonedVoices}
						disabled={loadingClonedVoices}
						class="text-sm text-purple-600 hover:text-purple-700 dark:text-purple-400 flex items-center gap-2"
					>
						<CarbonRenew class="size-4 {loadingClonedVoices ? 'animate-spin' : ''}" />
						Refresh
					</button>
				</div>

				{#if loadingClonedVoices}
					<div class="flex items-center justify-center py-8 text-gray-500 dark:text-gray-400">
						<CarbonRenew class="size-5 animate-spin mr-2" />
						Loading voices...
					</div>
				{:else if clonedVoices.length === 0}
					<div class="text-center py-8 text-gray-500 dark:text-gray-400">
						<CarbonMicrophone class="size-8 mx-auto mb-2 opacity-50" />
						<p>No voices yet</p>
						<p class="text-sm mt-1">Clone a voice, design one from description, or use premium speakers</p>
					</div>
				{:else}
					<div class="space-y-3">
						{#each clonedVoices as voice}
							<div class="flex items-center justify-between rounded-lg border {voice.voice_id === activeQwen3Voice ? 'border-purple-400 dark:border-purple-500 bg-purple-50 dark:bg-purple-900/20' : 'border-gray-200 dark:border-gray-700'} px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-800/50">
								<div class="flex-1">
									<div class="flex items-center gap-2">
										<span class="font-medium">{voice.voice_id}</span>
										{#if voice.voice_id === activeQwen3Voice}
											<span class="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-xs font-semibold text-purple-700 dark:bg-purple-900/40 dark:text-purple-300">
												<span class="size-1.5 rounded-full bg-purple-500 inline-block"></span>
												Active
											</span>
										{/if}
										<!-- Source Badge -->
										{#if voice.source === 'designed'}
											<span class="inline-flex items-center rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300">
												Designed
											</span>
										{:else}
											<span class="inline-flex items-center rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700 dark:bg-purple-900/40 dark:text-purple-300">
												Cloned
											</span>
										{/if}
									</div>
									<div class="text-sm text-gray-600 dark:text-gray-400">
										{voice.duration.toFixed(1)}s sample • {voice.sample_rate}Hz
										{#if voice.has_ref_text}
											• <span class="text-purple-600 dark:text-purple-400">ICL mode</span>
										{/if}
									</div>
								</div>
								<div class="flex items-center gap-1">
									{#if voice.voice_id !== activeQwen3Voice}
										<button
											onclick={() => setActiveVoice(voice.voice_id)}
											class="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-purple-600 hover:bg-purple-100 dark:text-purple-400 dark:hover:bg-purple-900/30"
											title="Set as active voice"
										>
											<CarbonCheckmark class="size-4" />
											Set Active
										</button>
									{/if}
									<button
										onclick={() => deleteClonedVoice(voice.voice_id)}
										class="text-red-600 hover:text-red-700 dark:text-red-400 p-2"
										title="Delete voice"
									>
										<CarbonTrash class="size-5" />
									</button>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>

			<!-- Test Synthesis -->
			{#if clonedVoices.length > 0}
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-5">
					<div class="rounded-lg bg-green-100 p-3 text-green-600 dark:bg-green-900/30 dark:text-green-400">
						<CarbonPlay class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Test Synthesis</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Preview how a cloned voice sounds on custom text
						</p>
					</div>
				</div>

				<div class="space-y-4">
					<!-- Voice Selector -->
					<div>
						<label for="synthesis-voice" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Voice
						</label>
						<select
							id="synthesis-voice"
							bind:value={synthesisTest.voice_id}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						>
							<option value="">Select a voice...</option>
							{#each clonedVoices as voice}
								<option value={voice.voice_id}>
									{voice.voice_id} ({voice.duration.toFixed(1)}s{voice.has_ref_text ? ' · ICL' : ''})
								</option>
							{/each}
						</select>
					</div>

					<!-- Text Input -->
					<div>
						<label for="synthesis-text" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Text to Speak
						</label>
						<textarea
							id="synthesis-text"
							bind:value={synthesisTest.text}
							placeholder="Enter text to synthesize with the cloned voice..."
							rows="3"
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
						></textarea>
					</div>

					<!-- Actions Row -->
					<div class="flex items-center justify-between gap-4">
						<!-- Audio Player (shows when audio is ready) -->
						{#if synthesisTest.audioUrl}
							<audio
								src={synthesisTest.audioUrl}
								controls
								class="flex-1 h-10"
							></audio>
						{:else}
							<div class="flex-1 text-sm text-gray-500 dark:text-gray-400">
								{synthesisTest.synthesizing ? 'Generating audio — this may take a moment on GPU...' : 'Audio will appear here after synthesis'}
							</div>
						{/if}

						<button
							onclick={testSynthesisQwen3}
							disabled={synthesisTest.synthesizing || !synthesisTest.voice_id || !synthesisTest.text.trim()}
							class="flex-shrink-0 rounded-lg bg-green-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50 dark:bg-green-500 dark:hover:bg-green-600"
						>
							<div class="flex items-center gap-2">
								{#if synthesisTest.synthesizing}
									<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
									Synthesizing...
								{:else}
									<CarbonPlay class="size-4" />
									Synthesize
								{/if}
							</div>
						</button>
					</div>
				</div>
			</div>
			{/if}

			<!-- Qwen3 Performance Settings -->
			{#if currentEngine === 'qwen3'}
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-6">
					<div class="rounded-lg bg-purple-100 p-3 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
						<CarbonSettings class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Performance Settings</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Tune synthesis speed and memory usage
						</p>
					</div>
					{#if qwen3Config.model_loaded}
						<span class="ml-auto inline-flex items-center gap-1.5 rounded-full bg-green-100 px-2.5 py-1 text-xs font-semibold text-green-700 dark:bg-green-900/40 dark:text-green-300">
							<span class="size-1.5 rounded-full bg-green-500 inline-block"></span>
							Model Loaded
						</span>
					{/if}
				</div>

				<div class="space-y-6">
					<!-- Chunk Size -->
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Chunk Size: {qwen3Config.chunk_size} characters
						</label>
						<input
							type="range"
							min="50"
							max="500"
							step="25"
							bind:value={qwen3Config.chunk_size}
							class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-purple-600"
						/>
						<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
							<span>50 (fastest)</span>
							<span>200 (balanced)</span>
							<span>500 (smoothest)</span>
						</div>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
							Smaller chunks = faster time-to-first-audio but more overhead. Larger chunks = smoother audio but slower start.
						</p>
					</div>

					<!-- Auto-unload toggle -->
					<div class="flex items-center justify-between">
						<div>
							<label class="text-sm font-medium text-gray-700 dark:text-gray-300">
								Auto-unload Model
							</label>
							<p class="text-xs text-gray-500 dark:text-gray-400">
								Free GPU memory after {qwen3Config.unload_timeout_seconds / 60} minutes of inactivity
							</p>
						</div>
						<button
							onclick={() => { qwen3Config.auto_unload = !qwen3Config.auto_unload; }}
							class="relative inline-flex h-6 w-11 items-center rounded-full transition-colors
								{qwen3Config.auto_unload ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}"
							role="switch"
							aria-checked={qwen3Config.auto_unload}
							aria-label="Toggle auto-unload model"
						>
							<span
								class="inline-block h-4 w-4 transform rounded-full bg-white transition-transform
									{qwen3Config.auto_unload ? 'translate-x-6' : 'translate-x-1'}"
							></span>
						</button>
					</div>

					<!-- Save Button -->
					<div class="flex justify-end pt-2">
						<button
							onclick={saveQwen3Config}
							disabled={savingQwen3Config}
							class="flex items-center gap-2 rounded-lg bg-purple-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 dark:bg-purple-500 dark:hover:bg-purple-600"
						>
							{#if savingQwen3Config}
								<CarbonRenew class="size-4 animate-spin" />
								Saving...
							{:else}
								<CarbonSave class="size-4" />
								Save Settings
							{/if}
						</button>
					</div>
				</div>
			</div>
			{/if}

			<!-- Qwen3 Info Box -->
			<div class="rounded-lg border border-purple-200 bg-purple-50 p-4 dark:border-purple-800 dark:bg-purple-900/20">
				<div class="flex items-start gap-3">
					<CarbonWarning class="size-5 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
					<div class="flex-1">
						<h3 class="font-semibold text-purple-900 dark:text-purple-200 text-sm">
							About Qwen3-TTS Voice Cloning
						</h3>
						<p class="text-sm text-purple-800 dark:text-purple-300 mt-1">
							Qwen3-TTS requires GPU with CUDA support. Voice cloning quality improves with clear audio (3-15s) and reference text.
							Synthesis is slower than Piper (~11-12x real-time on A100 GPU) but produces higher quality, personalized voices.
							Use this for important messages where voice consistency matters.
						</p>
					</div>
				</div>
			</div>
			{/if}
		</div>

	{:else if activeSection === 'test'}
		<!-- Test & Diagnostics -->
		<div class="space-y-6">

			<!-- System Health Overview -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center justify-between mb-6">
					<div>
						<h2 class="text-lg font-semibold">System Health</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Live status of all speech services
						</p>
					</div>
					<button
						onclick={checkHealth}
						disabled={loading}
						class="flex items-center gap-2 rounded-lg border border-gray-300 dark:border-gray-600 px-4 py-2 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
					>
						<CarbonRenew class="size-4 {loading ? 'animate-spin' : ''}" />
						Refresh
					</button>
				</div>

				{#if loading && !healthStatus}
					<div class="flex items-center justify-center py-12 text-gray-500 dark:text-gray-400">
						<div class="size-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent mr-3"></div>
						Checking services...
					</div>
				{:else if healthStatus}
					<!-- Overall Status Banner -->
					<div class="mb-5 flex items-center gap-3 rounded-lg px-4 py-3
						{healthStatus.status === 'ready'
							? 'bg-green-50 border border-green-200 dark:bg-green-900/20 dark:border-green-800'
							: healthStatus.status === 'degraded'
							? 'bg-yellow-50 border border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800'
							: 'bg-red-50 border border-red-200 dark:bg-red-900/20 dark:border-red-800'}"
					>
						<div class="size-2.5 rounded-full
							{healthStatus.status === 'ready' ? 'bg-green-500'
							: healthStatus.status === 'degraded' ? 'bg-yellow-500'
							: 'bg-red-500'}">
						</div>
						<span class="text-sm font-medium
							{healthStatus.status === 'ready' ? 'text-green-800 dark:text-green-200'
							: healthStatus.status === 'degraded' ? 'text-yellow-800 dark:text-yellow-200'
							: 'text-red-800 dark:text-red-200'}">
							System is {healthStatus.status === 'ready' ? 'fully operational' : healthStatus.status}
						</span>
					</div>

					<!-- Service Cards Grid -->
					<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
						<!-- Whisper STT -->
						<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
							<div class="flex items-start justify-between">
								<div class="flex items-center gap-3">
									<div class="rounded-lg bg-blue-100 p-2 dark:bg-blue-900/30">
										<CarbonMicrophone class="size-5 text-blue-600 dark:text-blue-400" />
									</div>
									<div>
										<p class="font-medium text-sm">Whisper STT</p>
										<p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
											{healthStatus.whisper_available ? 'Speech-to-text ready' : 'Not available'}
										</p>
									</div>
								</div>
								<div class="flex flex-col items-end gap-1">
									<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium
										{healthStatus.whisper_available
											? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
											: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}">
										<div class="size-1.5 rounded-full {healthStatus.whisper_available ? 'bg-green-500' : 'bg-red-500'}"></div>
										{healthStatus.whisper_available ? 'Available' : 'Unavailable'}
									</span>
									{#if healthStatus.whisper_loaded}
										<span class="text-xs text-blue-600 dark:text-blue-400">Model loaded</span>
									{:else}
										<span class="text-xs text-gray-400">Loads on demand</span>
									{/if}
								</div>
							</div>
							{#if healthStatus.whisper_config && Object.keys(healthStatus.whisper_config).length > 0}
								<div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 grid grid-cols-2 gap-x-4 gap-y-1">
									<div class="text-xs text-gray-500 dark:text-gray-400">Model</div>
									<div class="text-xs font-medium">{healthStatus.whisper_config.model_size || '—'}</div>
									<div class="text-xs text-gray-500 dark:text-gray-400">Device</div>
									<div class="text-xs font-medium">{healthStatus.whisper_config.device || '—'}</div>
									<div class="text-xs text-gray-500 dark:text-gray-400">Precision</div>
									<div class="text-xs font-medium">{healthStatus.whisper_config.compute_type || '—'}</div>
								</div>
							{/if}
						</div>

						<!-- Active TTS Engine -->
						<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
							<div class="flex items-start justify-between">
								<div class="flex items-center gap-3">
									<div class="rounded-lg p-2
										{healthStatus.tts_engine === 'qwen3'
											? 'bg-purple-100 dark:bg-purple-900/30'
											: 'bg-indigo-100 dark:bg-indigo-900/30'}">
										<CarbonTextToSpeech class="size-5
											{healthStatus.tts_engine === 'qwen3'
												? 'text-purple-600 dark:text-purple-400'
												: 'text-indigo-600 dark:text-indigo-400'}" />
									</div>
									<div>
										<p class="font-medium text-sm">
											{healthStatus.tts_engine === 'qwen3' ? 'Qwen3-TTS' : 'Piper TTS'}
										</p>
										<p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
											{healthStatus.tts_available ? 'Text-to-speech active' : 'Not available'}
										</p>
									</div>
								</div>
								<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium
									{healthStatus.tts_available
										? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
										: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}">
									<div class="size-1.5 rounded-full {healthStatus.tts_available ? 'bg-green-500' : 'bg-red-500'}"></div>
									{healthStatus.tts_available ? 'Active' : 'Unavailable'}
								</span>
							</div>
							<div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 grid grid-cols-2 gap-x-4 gap-y-1">
								<div class="text-xs text-gray-500 dark:text-gray-400">Engine</div>
								<div class="text-xs font-medium capitalize">{healthStatus.tts_engine || 'piper'}</div>
								<div class="text-xs text-gray-500 dark:text-gray-400">Mode</div>
								<div class="text-xs font-medium">
									{healthStatus.tts_engine === 'qwen3' ? 'Voice cloning' : 'Pre-built voices'}
								</div>
							</div>
						</div>

						<!-- Qwen3 Voice Cloning -->
						<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
							<div class="flex items-start justify-between">
								<div class="flex items-center gap-3">
									<div class="rounded-lg bg-purple-100 p-2 dark:bg-purple-900/30">
										<CarbonRecordingFilled class="size-5 text-purple-600 dark:text-purple-400" />
									</div>
									<div>
										<p class="font-medium text-sm">Qwen3 Voice Cloning</p>
										<p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
											{healthStatus.qwen3_available ? 'GPU available' : 'Requires GPU'}
										</p>
									</div>
								</div>
								<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium
									{healthStatus.qwen3_available
										? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
										: 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'}">
									<div class="size-1.5 rounded-full {healthStatus.qwen3_available ? 'bg-purple-500' : 'bg-gray-400'}"></div>
									{healthStatus.qwen3_available ? 'Available' : 'Disabled'}
								</span>
							</div>
							<div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 grid grid-cols-2 gap-x-4 gap-y-1">
								<div class="text-xs text-gray-500 dark:text-gray-400">Model</div>
								<div class="text-xs font-medium">{healthStatus.qwen3_loaded ? 'Loaded in GPU' : 'Not loaded'}</div>
								<div class="text-xs text-gray-500 dark:text-gray-400">Voices</div>
								<div class="text-xs font-medium">{clonedVoices.length} cloned</div>
							</div>
						</div>

						<!-- GPU Status (from /api/status) -->
						<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
							<div class="flex items-start justify-between">
								<div class="flex items-center gap-3">
									<div class="rounded-lg bg-orange-100 p-2 dark:bg-orange-900/30">
										<CarbonWarning class="size-5 text-orange-600 dark:text-orange-400" />
									</div>
									<div>
										<p class="font-medium text-sm">Hardware</p>
										<p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
											Compute resources
										</p>
									</div>
								</div>
							</div>
							<div class="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 grid grid-cols-2 gap-x-4 gap-y-1">
								<div class="text-xs text-gray-500 dark:text-gray-400">GPU</div>
								<div class="text-xs font-medium">
									{healthStatus.qwen3_available ? 'CUDA available' : 'CPU only'}
								</div>
								<div class="text-xs text-gray-500 dark:text-gray-400">Qwen3 engine</div>
								<div class="text-xs font-medium capitalize">
									{healthStatus.tts_engine === 'qwen3' ? 'Active' : 'Standby'}
								</div>
							</div>
						</div>
					</div>
				{:else}
					<div class="text-center py-12 text-gray-500 dark:text-gray-400">
						<CarbonWarning class="size-8 mx-auto mb-2 opacity-50" />
						<p>Could not reach backend</p>
						<p class="text-sm mt-1">Check that the backend is running on port 8000</p>
					</div>
				{/if}
			</div>

			<!-- STT Test -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-5">
					<div class="rounded-lg bg-blue-100 p-3 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
						<CarbonMicrophone class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Test Speech-to-Text</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							Upload an audio file to verify Whisper transcription
						</p>
					</div>
				</div>

				<div class="space-y-4">
					<div>
						<label for="diag-stt-file" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
							Audio File
						</label>
						<input
							id="diag-stt-file"
							type="file"
							accept="audio/*,.webm,.wav,.mp3,.m4a,.ogg,.flac"
							onchange={handleFileSelect}
							class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-900 file:mr-4 file:rounded file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 dark:file:bg-blue-900/30 dark:file:text-blue-400"
						/>
						{#if testAudioFile}
							<p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
								{testAudioFile.name} · {(testAudioFile.size / 1024).toFixed(1)} KB
							</p>
						{/if}
					</div>

					<button
						onclick={testTranscribe}
						disabled={!testAudioFile || testing}
						class="w-full rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
					>
						<div class="flex items-center justify-center gap-2">
							{#if testing}
								<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
								Transcribing...
							{:else}
								<CarbonPlay class="size-4" />
								Run Transcription Test
							{/if}
						</div>
					</button>

					{#if testTranscription}
						<div class="rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
							<p class="text-xs font-semibold text-blue-700 dark:text-blue-300 uppercase tracking-wide mb-2">
								Transcription Output
							</p>
							<p class="text-sm text-blue-900 dark:text-blue-100 whitespace-pre-wrap leading-relaxed">
								{testTranscription}
							</p>
						</div>
					{/if}
				</div>
			</div>

			<!-- TTS Quick Test -->
			<div class="rounded-xl border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
				<div class="flex items-center gap-3 mb-5">
					<div class="rounded-lg p-3
						{currentEngine === 'qwen3'
							? 'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400'
							: 'bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400'}">
						<CarbonTextToSpeech class="size-6" />
					</div>
					<div>
						<h2 class="text-lg font-semibold">Test Text-to-Speech</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400">
							{currentEngine === 'qwen3'
								? 'Quick synthesis test with current Qwen3 voice'
								: 'Quick synthesis test with current Piper voice'}
						</p>
					</div>
					<span class="ml-auto rounded-full px-2.5 py-1 text-xs font-medium
						{currentEngine === 'qwen3'
							? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
							: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'}">
						{currentEngine === 'qwen3' ? 'Qwen3' : 'Piper'}
					</span>
				</div>

				{#if currentEngine === 'qwen3' && clonedVoices.length === 0}
					<div class="rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20 text-sm text-yellow-800 dark:text-yellow-200">
						No cloned voices available. Switch to the TTS tab and clone a voice first.
					</div>
				{:else}
					<div class="space-y-4">
						{#if currentEngine === 'qwen3'}
							<div>
								<label for="diag-voice" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Voice</label>
								<select
									id="diag-voice"
									bind:value={synthesisTest.voice_id}
									class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
								>
									<option value="">Select a voice...</option>
									{#each clonedVoices as voice}
										<option value={voice.voice_id}>{voice.voice_id} ({voice.duration.toFixed(1)}s)</option>
									{/each}
								</select>
							</div>
						{/if}

						<div>
							<label for="diag-tts-text" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
								Text to Speak
							</label>
							<textarea
								id="diag-tts-text"
								bind:value={synthesisTest.text}
								placeholder="Hello! This is a test of the text-to-speech system."
								rows="2"
								class="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
							></textarea>
						</div>

						<div class="flex items-center gap-4">
							{#if synthesisTest.audioUrl}
								<audio src={synthesisTest.audioUrl} controls class="flex-1 h-10"></audio>
							{:else}
								<div class="flex-1 text-sm text-gray-500 dark:text-gray-400">
									{synthesisTest.synthesizing ? 'Generating audio...' : 'Audio output will appear here'}
								</div>
							{/if}

							<button
								onclick={diagTestTTS}
								disabled={synthesisTest.synthesizing || previewingVoice || !synthesisTest.text.trim() || (currentEngine === 'qwen3' && !synthesisTest.voice_id)}
								class="flex-shrink-0 rounded-lg px-5 py-2.5 text-sm font-medium text-white disabled:opacity-50
									{currentEngine === 'qwen3'
										? 'bg-purple-600 hover:bg-purple-700 dark:bg-purple-500 dark:hover:bg-purple-600'
										: 'bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600'}"
							>
								<div class="flex items-center gap-2">
									{#if synthesisTest.synthesizing || previewingVoice}
										<div class="size-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
										Generating...
									{:else}
										<CarbonPlay class="size-4" />
										Synthesize
									{/if}
								</div>
							</button>
						</div>
					</div>
				{/if}
			</div>

		</div>
	{/if}
</div>
