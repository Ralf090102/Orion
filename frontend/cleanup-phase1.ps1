# Phase 1 Cleanup Script - Remove Cloud-Specific Files
# Run this from the frontend directory

Write-Host "🧹 Starting Phase 1 Cleanup..." -ForegroundColor Cyan
Write-Host ""

$itemsToDelete = @(
    # Server-side logic (MongoDB, Auth)
    "src\lib\server",
    
    # Authentication routes
    "src\routes\login",
    "src\routes\logout",
    "src\routes\admin",
    
    # API routes (backend handles this)
    "src\routes\api",
    
    # Database migrations
    "src\lib\migrations",
    "src\lib\jobs",
    
    # Cloud features
    "src\routes\privacy",
    "src\routes\settings",
    "src\routes\models",
    "src\lib\components\mcp",
    
    # Cloud-specific components
    "src\lib\components\ShareConversationModal.svelte",
    "src\lib\components\BackgroundGenerationPoller.svelte",
    "src\lib\components\SubscribeModal.svelte",
    "src\lib\components\AnnouncementBanner.svelte",
    "src\lib\components\WelcomeModal.svelte"
)

$deletedCount = 0
$notFoundCount = 0

foreach ($item in $itemsToDelete) {
    if (Test-Path $item) {
        Write-Host "  ❌ Deleting: $item" -ForegroundColor Yellow
        Remove-Item -Path $item -Recurse -Force
        $deletedCount++
    } else {
        Write-Host "  ⏭️  Skipped (not found): $item" -ForegroundColor Gray
        $notFoundCount++
    }
}

Write-Host ""
Write-Host "✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "   - Deleted: $deletedCount items" -ForegroundColor Green
Write-Host "   - Not found: $notFoundCount items" -ForegroundColor Gray
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. npm install (to ensure dependencies are clean)"
Write-Host "   2. npm run dev (to test the frontend)"
Write-Host "   3. Make sure backend is running: uvicorn backend.app:app --reload --port 8000"
Write-Host ""
