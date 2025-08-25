import { writable, derived } from 'svelte/store';
import type { SystemConfig, GPUInfo } from '$lib/types/api';

// System configuration store
export const systemConfig = writable<SystemConfig | null>(null);

// GPU information store
export const gpuInfo = writable<GPUInfo | null>(null);

// Current profile store
export const currentProfile = writable<string>('default');

// Available profiles store
export const availableProfiles = writable<string[]>(['default']);

// Loading states
export const isLoadingConfig = writable<boolean>(false);
export const isLoadingGPU = writable<boolean>(false);

// System health store
export const systemHealth = writable<{
  status: string;
  timestamp: string;
  services: Record<string, boolean>;
} | null>(null);

// Derived store for system status
export const systemStatus = derived(
  [systemConfig, gpuInfo, systemHealth],
  ([$config, $gpu, $health]) => ({
    configured: $config !== null,
    gpuAvailable: $gpu?.available || false,
    healthy: $health?.status === 'healthy',
    services: $health?.services || {}
  })
);
