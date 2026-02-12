# Testing Guide - MindSync Training Service

Panduan lengkap untuk testing training service.

## 🧪 Quick Test

### Option 1: Automated Test Script (Recommended)

```bash
cd mindsync-model-training
python test_training.py
```

**Expected Output**:
```
🧪 MindSync Training Service - Test Suite
============================================================

TEST 1: Checking imports...
✅ wandb imported successfully
✅ scikit-learn imported successfully
✅ pandas imported successfully
✅ Custom Ridge Regression imported successfully

TEST 2: Checking dataset...
✅ Dataset found
✅ Dataset loaded: 5000 rows, 15 columns

...

📊 TEST SUMMARY
Results: 6/6 tests passed
🎉 All tests passed! Training service is ready.
```

### Option 2: Manual Quick Test

```bash
# Skip W&B upload for quick local test
cd mindsync-model-training
set SKIP_WANDB=true
python train.py
```

## 📋 Test Checklist

### 1. ✅ Environment Setup

```bash
# Check Python version
python --version
# Should be: Python 3.11.x or higher

# Check virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. ✅ Dataset Validation

```bash
# Check if dataset exists
ls df/ScreenTime\ vs\ MentalWellness.csv

# Quick data inspection
python -c "
import pandas as pd
df = pd.read_csv('df/ScreenTime vs MentalWellness.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Missing values: {df.isnull().sum().sum()}')
"
```

**Expected Output**:
```
Shape: (5000, 15)
Columns: ['age', 'gender', 'occupation', ...]
Missing values: 0
```

### 3. ✅ Custom Ridge Model

```bash
# Test custom model
python -c "
from custom_ridge import LinearRegressionRidge
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randn(100)

model = LinearRegressionRidge(alpha=1.0)
model.fit(X, y)
score = model.score(X, y)

print(f'Model trained successfully!')
print(f'R² score: {score:.4f}')
"
```

### 4. ✅ Training Pipeline (Local)

```bash
# Run training WITHOUT W&B upload (fast test)
set SKIP_WANDB=true
python train.py
```

**Expected Output**:
```
🚀 MindSync Model Training Pipeline
============================================================
📂 Loading dataset...
✅ Loaded 5000 samples
🔧 Creating preprocessing pipeline...
🎯 Training model...
✅ Training R²: 0.9234
✅ Test R²: 0.8876
✅ Test RMSE: 3.4521
🔍 Performing K-Means clustering...
📊 Extracting feature importance...
💾 Saving artifacts to artifacts/...
✅ Artifacts saved locally
⏭️ Skipping W&B upload (SKIP_WANDB=true)
✅ Training pipeline completed successfully!
```

### 5. ✅ Artifacts Generation

```bash
# Check if artifacts were created
ls artifacts/

# Should see:
# - model.pkl
# - preprocessor.pkl
# - healthy_cluster_avg.csv
# - model_coefficients.csv
# - feature_importance.csv
```

### 6. ✅ W&B Integration (Optional)

```bash
# Test W&B connection
wandb login
# (paste your API key)

# Test project access
wandb artifact list mindsync-model/mindsync-model

# Run full training with W&B upload
python train.py
```

**Expected Output**:
```
☁️ Uploading artifacts to Weights & Biases...
✅ Artifacts uploaded to W&B project: mindsync-model
📊 View your run at: https://wandb.ai/...
```

## 🐳 Docker Testing

### Build Docker Image

```bash
cd mindsync-model-training
docker build -t mindsync-training:test .
```

**Expected Output**:
```
Successfully built abc123def456
Successfully tagged mindsync-training:test
```

### Test Docker Container (Local Mode)

```bash
# Run without W&B upload
docker run -e SKIP_WANDB=true mindsync-training:test
```

### Test Docker Container (With W&B)

```bash
# Run with W&B upload
docker run \
  -e WANDB_API_KEY=your-api-key \
  -e WANDB_PROJECT=mindsync-model \
  -e WANDB_ENTITY=your-username \
  mindsync-training:test
```

## 🔍 Validation Tests

### Test 1: Model Performance

```python
# test_model_performance.py
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Load artifacts
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
df = pd.read_csv('df/ScreenTime vs MentalWellness.csv')
X_test = df.drop(['mental_wellness_score', 'user_id'], axis=1).head(100)
y_test = df['mental_wellness_score'].head(100)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Validation
assert r2 > 0.5, "R² too low"
assert rmse < 10, "RMSE too high"
print("✅ Model performance validation passed!")
```

### Test 2: Artifact Integrity

```python
# test_artifacts.py
import os
import pickle
import pandas as pd

artifacts_dir = 'artifacts'
required_files = [
    'model.pkl',
    'preprocessor.pkl',
    'healthy_cluster_avg.csv',
    'model_coefficients.csv',
    'feature_importance.csv'
]

print("Checking artifacts...")

for file in required_files:
    filepath = os.path.join(artifacts_dir, file)
    assert os.path.exists(filepath), f"Missing: {file}"
    
    # Check file is not empty
    assert os.path.getsize(filepath) > 0, f"Empty file: {file}"
    
    print(f"✅ {file}")

print("\n✅ All artifacts present and valid!")
```

### Test 3: Prediction Pipeline

```python
# test_prediction.py
import pickle
import pandas as pd

# Load model and preprocessor
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Test data
test_input = pd.DataFrame([{
    'age': 25,
    'gender': 'Male',
    'occupation': 'Student',
    'platform': 'Instagram',
    'screentime_hours': 5.5,
    'likes_per_day': 50,
    'follows_per_day': 15,
    'comments_per_day': 8,
    'messages_per_day': 25,
    'posts_per_day': 2,
    'video_hours': 3.5,
    'gaming_hours': 1.0,
    'productivity_apps_hours': 0.5,
    'social_media_hours': 5.0
}])

# Predict
prediction = model.predict(test_input)
print(f"Prediction: {prediction[0]:.2f}")

# Validation
assert 0 <= prediction[0] <= 100, "Prediction out of range"
print("✅ Prediction pipeline working!")
```

## 📊 Performance Benchmarks

### Expected Training Time

| Dataset Size | Training Time | Memory Usage |
|-------------|---------------|--------------|
| 1K samples  | ~5 seconds    | ~200 MB      |
| 5K samples  | ~15 seconds   | ~500 MB      |
| 10K samples | ~30 seconds   | ~1 GB        |

### Expected Metrics

| Metric      | Good    | Acceptable | Poor   |
|-------------|---------|------------|--------|
| Train R²    | > 0.90  | > 0.80     | < 0.80 |
| Test R²     | > 0.85  | > 0.75     | < 0.75 |
| RMSE        | < 3.5   | < 5.0      | > 5.0  |
| MAE         | < 2.5   | < 4.0      | > 4.0  |

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'wandb'"

```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: df/ScreenTime..."

```bash
# Check if dataset exists
ls df/

# If missing, ensure you're in the right directory
cd mindsync-model-training
```

### Issue: Training is too slow

```bash
# Use smaller sample for testing
python -c "
import pandas as pd
df = pd.read_csv('df/ScreenTime vs MentalWellness.csv')
df.sample(n=1000, random_state=42).to_csv('df/sample.csv', index=False)
"

# Then modify train.py to use sample.csv temporarily
```

### Issue: W&B upload fails

```bash
# Skip W&B for local testing
set SKIP_WANDB=true
python train.py

# Or check W&B credentials
wandb login
wandb whoami
```

## ✅ Success Criteria

Training service is working correctly if:

- ✅ All tests in `test_training.py` pass
- ✅ Training completes without errors
- ✅ Test R² > 0.75
- ✅ All 5 artifact files generated
- ✅ Artifacts upload to W&B (if enabled)
- ✅ Model can make predictions on new data

## 🚀 Next Steps

After successful testing:

1. **Commit artifacts** (optional): Keep local copy
2. **Push to W&B**: Run with `SKIP_WANDB=false`
3. **Test inference service**: Use artifacts in mindsync-model-flask
4. **Setup CI/CD**: Configure GitHub Actions
5. **Schedule training**: Set up weekly/monthly retraining

## 📞 Need Help?

Run the automated test suite first:
```bash
python test_training.py
```

Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues.

---

**Last Updated**: 2026-02-12
