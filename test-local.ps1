# Quick Test Script - Training Service
# Run this to quickly test if training service works

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Training Service" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "train.py")) {
    Write-Host "❌ Error: train.py not found" -ForegroundColor Red
    Write-Host "   Make sure you're in mindsync-model-training directory" -ForegroundColor Yellow
    exit 1
}

# Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found" -ForegroundColor Yellow
    Write-Host "   Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "   ⚠️  IMPORTANT: Edit .env and add your WANDB_API_KEY" -ForegroundColor Red
    Write-Host "   Then run this script again" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🧪 Running Tests" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Run tests
python test_training.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Tests failed!" -ForegroundColor Red
    Write-Host "   Check the output above for errors" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "✅ All tests passed!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Ask if want to run full training
$response = Read-Host "Do you want to run full training and upload to W`&B? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "🚀 Running Full Training" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    
    python train.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=====================================" -ForegroundColor Green
        Write-Host "✅ Training completed successfully!" -ForegroundColor Green
        Write-Host "=====================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Check W`&B dashboard: https://wandb.ai" -ForegroundColor Yellow
        Write-Host "2. Verify artifacts uploaded" -ForegroundColor Yellow
        Write-Host "3. Test inference service with new model" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "❌ Training failed!" -ForegroundColor Red
        Write-Host "   Check WANDB_API_KEY in .env file" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "✅ Tests completed. Full training skipped." -ForegroundColor Green
    Write-Host ""
    Write-Host "To run full training manually:" -ForegroundColor Yellow
    Write-Host "  python train.py" -ForegroundColor Yellow
}
