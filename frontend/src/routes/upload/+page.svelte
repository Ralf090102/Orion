<script lang="ts">
	import Navigation from '$lib/components/Navigation.svelte';
	import { documents, uploadProgress, documentsStore } from '$lib/stores/documents';
	import { documentsApi } from '$lib/utils/api';
	import { Upload, FileText, Trash2, CheckCircle, AlertCircle, Loader2 } from 'lucide-svelte';
	
	let fileInput: HTMLInputElement;
	let isDragOver = false;
	
	const handleFileSelect = (files: FileList | null) => {
		if (!files) return;
		
		Array.from(files).forEach(file => {
			uploadFile(file);
		});
	};
	
	const uploadFile = async (file: File) => {
		documentsStore.addUploadProgress(file.name);
		
		try {
			documentsStore.updateUploadProgress(file.name, { status: 'uploading' });
			
			const document = await documentsApi.uploadDocument(file, (progress) => {
				documentsStore.updateUploadProgress(file.name, { progress });
			});
			
			documentsStore.updateUploadProgress(file.name, { 
				status: 'complete', 
				progress: 100 
			});
			
			documentsStore.addDocument(document);
			
			// Remove progress after 2 seconds
			setTimeout(() => {
				documentsStore.removeUploadProgress(file.name);
			}, 2000);
			
		} catch (error) {
			console.error('Upload failed:', error);
			documentsStore.updateUploadProgress(file.name, { 
				status: 'error', 
				error: 'Upload failed' 
			});
		}
	};
	
	const handleDrop = (event: DragEvent) => {
		event.preventDefault();
		isDragOver = false;
		
		if (event.dataTransfer?.files) {
			handleFileSelect(event.dataTransfer.files);
		}
	};
	
	const handleDragOver = (event: DragEvent) => {
		event.preventDefault();
		isDragOver = true;
	};
	
	const handleDragLeave = () => {
		isDragOver = false;
	};
	
	const deleteDocument = async (documentId: string) => {
		try {
			await documentsApi.deleteDocument(documentId);
			documentsStore.removeDocument(documentId);
		} catch (error) {
			console.error('Delete failed:', error);
		}
	};
	
	const formatFileSize = (bytes: number): string => {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	};
	
	const getStatusIcon = (status: string) => {
		switch (status) {
			case 'complete':
			case 'completed':
				return CheckCircle;
			case 'error':
			case 'failed':
				return AlertCircle;
			case 'pending':
			case 'processing':
				return Loader2;
			default:
				return FileText;
		}
	};
</script>

<Navigation />

<div class="container mx-auto max-w-4xl p-6">
	<div class="mb-8">
		<h1 class="text-3xl font-bold mb-2">Document Upload</h1>
		<p class="text-base-content/70">Upload documents to enhance your RAG assistant's knowledge base.</p>
	</div>
	
	<!-- Upload Area -->
	<div class="card bg-base-200 shadow-lg mb-8">
		<div class="card-body">
			<div
				class="upload-area"
				class:dragover={isDragOver}
				on:drop={handleDrop}
				on:dragover={handleDragOver}
				on:dragleave={handleDragLeave}
				role="button"
				tabindex="0"
				on:click={() => fileInput.click()}
				on:keydown={(e) => e.key === 'Enter' && fileInput.click()}>
				
				<Upload size={48} class="mx-auto mb-4 text-primary" />
				<p class="text-lg font-semibold mb-2">Drop files here or click to upload</p>
				<p class="text-sm text-base-content/60">
					Supported formats: PDF, DOCX, PPTX, XLSX, TXT, MD
				</p>
				
				<input
					bind:this={fileInput}
					type="file"
					multiple
					accept=".pdf,.docx,.pptx,.xlsx,.txt,.md"
					class="hidden"
					on:change={(e) => handleFileSelect(e.currentTarget.files)} />
			</div>
		</div>
	</div>
	
	<!-- Upload Progress -->
	{#if $uploadProgress.length > 0}
		<div class="card bg-base-100 shadow-lg mb-8">
			<div class="card-body">
				<h2 class="card-title">Upload Progress</h2>
				
				{#each $uploadProgress as progress}
					<div class="flex items-center gap-4 p-3 bg-base-200 rounded-lg">
						<div class="flex-1">
							<div class="font-medium">{progress.filename}</div>
							<div class="text-sm text-base-content/70">
								{progress.status === 'uploading' ? `${progress.progress}%` : progress.status}
								{#if progress.error}
									- {progress.error}
								{/if}
							</div>
						</div>
						
						{#if progress.status === 'uploading'}
							<div class="w-24">
								<progress class="progress progress-primary w-full" value={progress.progress} max="100"></progress>
							</div>
						{/if}
						
						<svelte:component 
							this={getStatusIcon(progress.status)}
							size={20}
							class={progress.status === 'error' ? 'text-error' : 
								   progress.status === 'complete' ? 'text-success' : 
								   progress.status === 'processing' ? 'text-warning animate-spin' : 'text-base-content'} />
					</div>
				{/each}
			</div>
		</div>
	{/if}
	
	<!-- Documents List -->
	<div class="card bg-base-100 shadow-lg">
		<div class="card-body">
			<h2 class="card-title">Uploaded Documents</h2>
			
			{#if $documents.length === 0}
				<div class="text-center py-8">
					<FileText size={48} class="mx-auto mb-4 text-base-content/30" />
					<p class="text-base-content/70">No documents uploaded yet.</p>
				</div>
			{:else}
				<div class="overflow-x-auto">
					<table class="table">
						<thead>
							<tr>
								<th>Name</th>
								<th>Size</th>
								<th>Uploaded</th>
								<th>Status</th>
								<th>Chunks</th>
								<th>Actions</th>
							</tr>
						</thead>
						<tbody>
							{#each $documents as doc}
								<tr>
									<td>
										<div class="flex items-center gap-3">
											<FileText size={16} />
											<div class="font-medium">{doc.filename}</div>
										</div>
									</td>
									<td>{formatFileSize(doc.file_size)}</td>
									<td>{new Date(doc.upload_date).toLocaleDateString()}</td>
									<td>
										<div class="flex items-center gap-2">
											<svelte:component 
												this={getStatusIcon(doc.processing_status)}
												size={16}
												class={doc.processing_status === 'failed' ? 'text-error' : 
													   doc.processing_status === 'completed' ? 'text-success' : 
													   'text-warning'} />
											<span class="capitalize">{doc.processing_status}</span>
										</div>
									</td>
									<td>{doc.chunk_count || 'N/A'}</td>
									<td>
										<button
											on:click={() => deleteDocument(doc.id)}
											class="btn btn-error btn-sm">
											<Trash2 size={14} />
										</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	</div>
</div>
