# MindSync Model Training Service

**Independent microservice** untuk training model machine learning MindSync dengan integrasi Weights & Biases (W&B).

> **Note**: Service ini adalah microservice yang terpisah dan independen dari inference service. Deployment dan workflow-nya tidak terhubung langsung dengan service lain.

## 🎯 Tujuan

Service ini bertanggung jawab untuk:
- Training model Ridge Regression untuk prediksi mental wellness
- Melakukan clustering untuk identifikasi pola healthy user
- Upload model artifacts ke Weights & Biases
- Versioning model untuk deployment ke inference service

## 🏗️ Architecture

```
Training Service (Standalone)
    ↓
Train Model
    ↓
Upload to W&B
    ↓
Inference Service downloads independently
```

## � CI/CD Workflow

### Automatic Training on Push to Main

```
Developer creates PR → Tests run → Merge to main
    ↓
GitHub Actions: train.yml auto-triggers
    ↓
Train model & upload to W&B ✅
```

**Triggers**:
- `push` to `main` branch (if .py or requirements.txt changed)
- `schedule` - Weekly on Sunday
- `workflow_dispatch` - Manual trigger

### Standard Development Flow

```bash
# 1. Create feature branch
git checkout -b feature/improve-model

# 2. Make changes to train.py, custom_ridge.py, etc.
# 3. Create PR and get review
# 4. Merge to main → Training auto-runs! ✅
```

## �📋 Prerequisites

- Python 3.11+
- Weights & Biases account ([daftar di sini](https://wandb.ai/))
- W&B API key

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Weights & Biases

Login ke W&B:

```bash
wandb login
```

Masukkan API key Anda ketika diminta.

### 3. Set Environment Variables

Buat file `.env`:

```bash
WANDB_PROJECT=mindsync-model
WANDB_ENTITY=your-wandb-username
MODEL_VERSION=v1.0
```

## 🏃 Menjalankan Training

### Local Training

```bash
python train.py
```

### Docker Training

Build image:

```bash
docker build -t mindsync-model-training .
```

Run training:

```bash
docker run -e WANDB_API_KEY=your-api-key \
           -e WANDB_PROJECT=mindsync-model \
           -e MODEL_VERSION=v1.0 \
           mindsync-model-training
```

## 📦 Artifacts yang Dihasilkan

Training akan menghasilkan artifacts berikut:

### Uploaded to W&B:
1. **model.pkl** - Trained Ridge Regression model ⬆️
2. **preprocessor.pkl** - Data preprocessing pipeline ⬆️
3. **model_coefficients.csv** - Koefisien model untuk interpretability ⬆️
4. **feature_importance.csv** - Top 20 fitur paling penting ⬆️

### Generated Locally (NOT uploaded):
5. **healthy_cluster_avg.csv** - Nilai rata-rata cluster healthy users 📌
   - **Not uploaded to W&B** (inference service maintains its own version)
   - Used for local validation only

## 🔄 CI/CD Integration

Service ini memiliki GitHub Actions workflow sendiri di `.github/workflows/`:

- **train.yml** - Training pipeline (manual atau scheduled)
- **test.yml** - Automated testing untuk PR dan commits

### Setup GitHub Actions

1. Add secrets di repository settings:
   - `WANDB_API_KEY` - Your W&B API key
   - `WANDB_ENTITY` - Your W&B username/team

2. Workflow akan otomatis run:
   - Manual trigger: Actions → Train Model → Run workflow
   - Scheduled: Setiap Sunday jam 00:00 UTC
   - PR/commits: Automatic tests

### Manual Trigger

```bash
# Via GitHub UI: Actions → Train Model → Run workflow
# Atau via gh CLI:
gh workflow run train.yml -f model_version=v1.1.0
```

## 📊 Model Performance

Model akan log metrics berikut ke W&B:
- Training R² Score
- Test R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## 🔍 Monitoring

Akses W&B dashboard untuk:
- Tracking training runs
- Compare model versions
- Visualize metrics over time
- Download artifacts

Link: `https://wandb.ai/YOUR_ENTITY/mindsync-model`

## 🛠️ Development

### Kustomisasi Model

Edit hyperparameters di [train.py](train.py):

```python
LinearRegressionRidge(
    alpha=1.0,          # Regularization strength
    solver="closed_form" # Solver method
)
```

### Testing Training Pipeline

Skip W&B upload untuk testing local:

```bash
SKIP_WANDB=true python train.py
```

## 📝 Dataset

Dataset: `df/ScreenTime vs MentalWellness.csv`

Features:
- **Numeric**: Age, screen time metrics, mental health indicators
- **Categorical**: Gender, occupation, platform

Target: `mental_wellness_score`

## 🔗 Integration dengan Inference Service

Setelah training selesai:
1. Model artifacts ter-upload ke W&B dengan version tag
2. Inference service (`mindsync-model-flask`) akan download artifacts terbaru
3. Automatic deployment melalui CI/CD pipeline

## 🤝 Contributing

1. Test training pipeline locally
2. Verify W&B upload berhasil
3. Check metrics di W&B dashboard
4. Pastikan inference service bisa download artifacts

## 📄 License

[Your License Here]
