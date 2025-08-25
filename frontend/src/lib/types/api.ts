// API Response Types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// System Configuration Types
export interface SystemConfig {
  max_cpu_usage_percent: number;
  max_memory_usage_gb: number;
  enable_gpu_acceleration: boolean;
  gpu_memory_limit_mb?: number;
  preferred_gpu_device?: number;
  auto_indexing: boolean;
  indexing_interval_minutes: number;
  watch_directories: string[];
  excluded_extensions: string[];
  chunk_size: number;
  chunk_overlap: number;
  vector_dimensions: number;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: DocumentSource[];
}

export interface DocumentSource {
  filename: string;
  page?: number;
  chunk_id: string;
  relevance_score: number;
  content_preview: string;
}

// Document Upload Types
export interface UploadProgress {
  filename: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'complete' | 'error';
  error?: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  file_size: number;
  upload_date: Date;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count?: number;
  content_type: string;
}

// GPU Information
export interface GPUInfo {
  available: boolean;
  device_count: number;
  current_device?: number;
  memory_total?: number;
  memory_used?: number;
  driver_version?: string;
}

// Query Types
export interface QueryRequest {
  query: string;
  profile?: string;
  max_results?: number;
  include_sources?: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: DocumentSource[];
  processing_time: number;
  tokens_used?: number;
}
