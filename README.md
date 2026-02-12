# MindSync Model Training Service

Service untuk training model machine learning MindSync dengan integrasi Weights & Biases (W&B).

## 🎯 Tujuan

Service ini bertanggung jawab untuk:
- Training model Ridge Regression untuk prediksi mental wellness
- Melakukan clustering untuk identifikasi pola healthy user
- Upload model artifacts ke Weights & Biases
- Versioning model untuk deployment ke inference service

## 📋 Prerequisites

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

Training akan menghasilkan artifacts berikut dan upload ke W&B:

1. **model.pkl** - Trained Ridge Regression model
2. **preprocessor.pkl** - Data preprocessing pipeline
3. **healthy_cluster_avg.csv** - Nilai rata-rata cluster healthy users
4. **model_coefficients.csv** - Koefisien model untuk interpretability
5. **feature_importance.csv** - Top 20 fitur paling penting

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Train Model

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly training
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and run training
        run: |
          cd mindsync-model-training
          docker build -t mindsync-training .
          docker run -e WANDB_API_KEY=${{ secrets.WANDB_API_KEY }} \
                     -e WANDB_PROJECT=mindsync-model \
                     mindsync-training
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
