import { writable } from 'svelte/store';
import type { DocumentInfo, UploadProgress } from '$lib/types/api';

// Documents store
export const documents = writable<DocumentInfo[]>([]);

// Upload progress store
export const uploadProgress = writable<UploadProgress[]>([]);

// Loading states
export const isLoadingDocuments = writable<boolean>(false);

// Upload helper functions
export const documentsStore = {
  addDocument: (document: DocumentInfo) => {
    documents.update(docs => [...docs, document]);
  },

  removeDocument: (documentId: string) => {
    documents.update(docs => docs.filter(doc => doc.id !== documentId));
  },

  updateDocument: (documentId: string, updates: Partial<DocumentInfo>) => {
    documents.update(docs => 
      docs.map(doc => doc.id === documentId ? { ...doc, ...updates } : doc)
    );
  },

  addUploadProgress: (filename: string) => {
    const progress: UploadProgress = {
      filename,
      progress: 0,
      status: 'pending'
    };
    uploadProgress.update(progresses => [...progresses, progress]);
  },

  updateUploadProgress: (filename: string, updates: Partial<UploadProgress>) => {
    uploadProgress.update(progresses =>
      progresses.map(p => p.filename === filename ? { ...p, ...updates } : p)
    );
  },

  removeUploadProgress: (filename: string) => {
    uploadProgress.update(progresses => 
      progresses.filter(p => p.filename !== filename)
    );
  },

  clearCompletedUploads: () => {
    uploadProgress.update(progresses =>
      progresses.filter(p => p.status !== 'complete')
    );
  }
};
