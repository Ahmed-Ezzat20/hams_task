# Windows Setup Script for Arabic Voice Agent
# Run this script in PowerShell to set up the environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Arabic Voice Agent - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.9 or higher." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists
if (Test-Path "hams") {
    Write-Host "✓ Virtual environment 'hams' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment 'hams'..." -ForegroundColor Yellow
    python -m venv hams
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\hams\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Install Arabic turn detector plugin
Write-Host ""
Write-Host "Installing Arabic turn detector plugin..." -ForegroundColor Yellow
Set-Location livekit_plugins_arabic_turn_detector
pip install -e .
Set-Location ..

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Arabic turn detector plugin installed" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install plugin" -ForegroundColor Red
    exit 1
}

# Check if .env.local exists
Write-Host ""
if (Test-Path ".env.local") {
    Write-Host "✓ .env.local already exists" -ForegroundColor Green
} else {
    Write-Host "Creating .env.local from template..." -ForegroundColor Yellow
    Copy-Item .env.local.example .env.local
    Write-Host "✓ .env.local created" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env.local with your API keys!" -ForegroundColor Red
    Write-Host "   Required variables:" -ForegroundColor Yellow
    Write-Host "   - LIVEKIT_URL" -ForegroundColor Yellow
    Write-Host "   - LIVEKIT_API_KEY" -ForegroundColor Yellow
    Write-Host "   - LIVEKIT_API_SECRET" -ForegroundColor Yellow
    Write-Host "   - ELEVENLABS_API_KEY" -ForegroundColor Yellow
    Write-Host "   - GOOGLE_API_KEY" -ForegroundColor Yellow
}

# Check if model exists
Write-Host ""
$modelPath = ".\eou_model\models\eou_model_quantized.onnx"
if (Test-Path $modelPath) {
    $modelSize = (Get-Item $modelPath).Length / 1MB
    Write-Host "✓ Arabic EOU model found: $([math]::Round($modelSize, 2)) MB" -ForegroundColor Green
} else {
    Write-Host "⚠️  Arabic EOU model not found at: $modelPath" -ForegroundColor Yellow
    Write-Host "   You need to train and quantize the model first." -ForegroundColor Yellow
    Write-Host "   See eou_model/README.md for instructions." -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env.local with your API keys" -ForegroundColor White
Write-Host "2. Ensure the EOU model is trained and quantized" -ForegroundColor White
Write-Host "3. Run the agent:" -ForegroundColor White
Write-Host "   python agent_with_arabic_turn_detector.py dev" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment in the future:" -ForegroundColor Yellow
Write-Host "   .\hams\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
