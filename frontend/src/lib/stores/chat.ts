import { writable } from 'svelte/store';
import type { ChatMessage } from '$lib/types/api';

// Chat messages store
export const chatMessages = writable<ChatMessage[]>([]);

// Current query store
export const currentQuery = writable<string>('');

// Loading state for chat
export const isChatLoading = writable<boolean>(false);

// Chat settings
export const chatSettings = writable({
  maxResults: 5,
  includeSources: true,
  autoScroll: true
});

// Helper functions
export const chatStore = {
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const newMessage: ChatMessage = {
      ...message,
      id: crypto.randomUUID(),
      timestamp: new Date()
    };
    
    chatMessages.update(messages => [...messages, newMessage]);
  },

  clearMessages: () => {
    chatMessages.set([]);
  },

  setLoading: (loading: boolean) => {
    isChatLoading.set(loading);
  }
};
