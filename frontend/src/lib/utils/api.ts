import axios, { type AxiosResponse } from 'axios';
import type { 
  APIResponse, 
  SystemConfig, 
  QueryRequest, 
  QueryResponse,
  GPUInfo,
  DocumentInfo
} from '$lib/types/api';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8003/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// System API
export const systemApi = {
  async getConfig(): Promise<SystemConfig> {
    const response: AxiosResponse<SystemConfig> = await api.get('/system/config');
    return response.data;
  },

  async updateConfig(config: Partial<SystemConfig>): Promise<SystemConfig> {
    const response: AxiosResponse<SystemConfig> = await api.put('/system/config', config);
    return response.data;
  },

  async getGPUInfo(): Promise<GPUInfo> {
    const response: AxiosResponse<GPUInfo> = await api.get('/system/gpu/capabilities');
    return response.data;
  },

  async getHealth(): Promise<{ status: string; timestamp: string; services: Record<string, boolean> }> {
    const response = await api.get('/system/health');
    return response.data;
  },

  async getProfiles(): Promise<string[]> {
    const response: AxiosResponse<string[]> = await api.get('/system/profiles');
    return response.data;
  },

  async switchProfile(profileName: string): Promise<{ message: string }> {
    const response = await api.post(`/system/profiles/${profileName}/switch`);
    return response.data;
  }
};

// Query API
export const queryApi = {
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response: AxiosResponse<QueryResponse> = await api.post('/query', request);
    return response.data;
  }
};

// Documents API  
export const documentsApi = {
  async getDocuments(): Promise<DocumentInfo[]> {
    const response: AxiosResponse<DocumentInfo[]> = await api.get('/documents');
    return response.data;
  },

  async uploadDocument(file: File, onProgress?: (progress: number) => void): Promise<DocumentInfo> {
    const formData = new FormData();
    formData.append('file', file);

    const response: AxiosResponse<DocumentInfo> = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(Math.round(progress));
        }
      },
    });

    return response.data;
  },

  async deleteDocument(documentId: string): Promise<{ message: string }> {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  }
};

// Export default api instance for custom requests
export default api;
