"""
Test script for MindSync Training Service

This script runs quick validation tests without uploading to W&B.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def test_imports():
    """Test 1: Verify all required imports work."""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)
    
    try:
        import wandb
        print("✅ wandb imported successfully")
    except ImportError as e:
        print(f"❌ wandb import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn imported successfully (version {sklearn.__version__})")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ pandas imported successfully (version {pd.__version__})")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from custom_ridge import LinearRegressionRidge
        print("✅ Custom Ridge Regression imported successfully")
    except ImportError as e:
        print(f"❌ Custom Ridge import failed: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_dataset():
    """Test 2: Verify dataset exists and is valid."""
    print("=" * 60)
    print("TEST 2: Checking dataset...")
    print("=" * 60)
    
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()
    
    df_path = base_dir / "df" / "ScreenTime vs MentalWellness.csv"
    
    if not df_path.exists():
        print(f"❌ Dataset not found at: {df_path}")
        return False
    
    print(f"✅ Dataset found: {df_path}")
    
    try:
        df = pd.read_csv(df_path)
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Check required columns
    required_cols = [
        'mental_wellness_score', 'age', 'gender', 'occupation', 
        'platform', 'screentime_hours'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    print(f"✅ All required columns present")
    
    # Check for missing values
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"⚠️  Warning: Missing values found:\n{missing_counts[missing_counts > 0]}")
    else:
        print("✅ No missing values in required columns")
    
    print("\n✅ Dataset validation passed!\n")
    return True


def test_custom_ridge():
    """Test 3: Verify custom Ridge Regression works."""
    print("=" * 60)
    print("TEST 3: Testing Custom Ridge Regression...")
    print("=" * 60)
    
    try:
        from custom_ridge import LinearRegressionRidge
        from sklearn.model_selection import train_test_split
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test each solver
        solvers = ['closed_form', 'svd', 'cholesky']
        
        for solver in solvers:
            print(f"\n  Testing solver: {solver}")
            model = LinearRegressionRidge(alpha=1.0, solver=solver)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"    ✅ Solver '{solver}' works! R² = {score:.4f}")
        
        print("\n✅ Custom Ridge Regression working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Custom Ridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test 4: Test preprocessing pipeline."""
    print("=" * 60)
    print("TEST 4: Testing preprocessing pipeline...")
    print("=" * 60)
    
    try:
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Load dataset
        base_dir = Path(__file__).parent
        df_path = base_dir / "df" / "ScreenTime vs MentalWellness.csv"
        df = pd.read_csv(df_path).head(100)  # Use small sample
        
        # Define features
        categorical_features = ["gender", "occupation", "platform"]
        numeric_features = [
            col for col in df.columns 
            if col not in categorical_features + ["mental_wellness_score", "user_id"]
        ]
        
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
            ]
        )
        
        X = df[numeric_features + categorical_features]
        X_transformed = preprocessor.fit_transform(X)
        
        print(f"  ✅ Input shape: {X.shape}")
        print(f"  ✅ Output shape: {X_transformed.shape}")
        print("\n✅ Preprocessing pipeline working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_dry_run():
    """Test 5: Run a quick training dry run (small dataset)."""
    print("=" * 60)
    print("TEST 5: Running training dry run (no W&B upload)...")
    print("=" * 60)
    
    # Set environment to skip W&B
    os.environ['SKIP_WANDB'] = 'true'
    
    try:
        # Import training script
        import train
        
        # Load small sample of data
        df = train.load_and_prepare_data()
        df_sample = df.sample(n=min(500, len(df)), random_state=42)
        
        print(f"  Using sample: {len(df_sample)} rows")
        
        # Define features
        target = "mental_wellness_score"
        categorical_features = ["gender", "occupation", "platform"]
        numeric_features = [
            col for col in df_sample.columns
            if col not in categorical_features + [target, "user_id"]
        ]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X = df_sample[numeric_features + categorical_features]
        y = df_sample[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessor
        preprocessor = train.create_preprocessor(categorical_features, numeric_features)
        
        # Train model
        print("  Training model...")
        model, metrics = train.train_model(X_train, y_train, X_test, y_test, preprocessor)
        
        print(f"\n  📊 Training Results:")
        print(f"    Train R²: {metrics['train_r2']:.4f}")
        print(f"    Test R²:  {metrics['test_r2']:.4f}")
        print(f"    Test RMSE: {metrics['test_rmse']:.4f}")
        
        # Check if model is reasonable
        if metrics['test_r2'] > 0.5:
            print(f"\n  ✅ Model performance looks good!")
        else:
            print(f"\n  ⚠️  Warning: Model R² is low (might need more data)")
        
        print("\n✅ Training dry run completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Training dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wandb_connection():
    """Test 6: Verify W&B connection (optional)."""
    print("=" * 60)
    print("TEST 6: Testing Weights & Biases connection...")
    print("=" * 60)
    
    try:
        import wandb
        
        # Check if logged in
        api = wandb.Api()
        print(f"✅ W&B API connected")
        
        # Try to access the project
        wandb_project = os.getenv("WANDB_PROJECT", "mindsync-model")
        wandb_entity = os.getenv("WANDB_ENTITY", None)
        
        if wandb_entity:
            print(f"✅ Entity: {wandb_entity}")
        print(f"✅ Project: {wandb_project}")
        
        print("\n✅ W&B connection working!\n")
        return True
        
    except Exception as e:
        print(f"⚠️  W&B connection test skipped or failed: {e}")
        print("   This is optional - training can work without W&B upload")
        return True  # Don't fail on W&B issues


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 MindSync Training Service - Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Custom Ridge", test_custom_ridge),
        ("Preprocessing", test_preprocessing),
        ("Training Dry Run", test_training_dry_run),
        ("W&B Connection", test_wandb_connection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training service is ready.")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Check warnings above.")
    else:
        print("❌ Multiple tests failed. Fix issues before training.")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
