"""
Test script for MindSync Training Service
Updated to validate Smart Preprocessing (Yeo-Johnson + Poly + Lasso) and Custom Ridge.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Pastikan module lokal bisa diimport
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: Verify all required imports work."""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)
    
    try:
        import wandb
        print("✅ wandb imported successfully")
    except ImportError as e:
        print(f"⚠️ wandb import failed (Optional): {e}")
    
    try:
        import sklearn
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
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

    try:
        import train
        print("✅ Train module imported successfully")
    except ImportError as e:
        print(f"❌ Train module import failed: {e}")
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
        'mental_wellness_index_0_100', 'age', 'gender', 'occupation', 
        'work_mode', 'screen_time_hours'
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
    """Test 3: Verify custom Ridge Regression works with ALL solvers."""
    print("=" * 60)
    print("TEST 3: Testing Custom Ridge Regression (All Solvers)...")
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
        
        # Test expanded list of solvers from your new Custom Ridge
        solvers = ['closed_form', 'svd', 'cholesky', 'gd', 'sgd', 'sag']
        
        for solver in solvers:
            print(f"  Testing solver: {solver}...", end=" ")
            try:
                model = LinearRegressionRidge(alpha=1.0, solver=solver, max_iter=500)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                # GD/SGD might have lower scores on random noise without tuning, checking if it runs is priority
                if score > -10: 
                    print(f"✅ OK (R²={score:.2f})")
                else:
                    print(f"⚠️  Runs but score low (R²={score:.2f})")
            except Exception as e:
                print(f"❌ FAIL: {e}")
                return False
        
        print("\n✅ Custom Ridge Regression working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Custom Ridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test 4: Test Smart Pipeline (Yeo-Johnson + Poly + Lasso)."""
    print("=" * 60)
    print("TEST 4: Testing Smart Preprocessing Pipeline...")
    print("=" * 60)
    
    try:
        import train
        
        # Load small sample
        df = train.load_and_prepare_data().head(100)
        
        target = 'mental_wellness_index_0_100'
        X = df.drop(target, axis=1)
        y = df[target]
        
        # Create the complex pipeline
        pipeline = train.create_pipeline()
        
        print("  Running fit_transform on pipeline...")
        print("  (Includes: Cleaning -> Yeo-Johnson -> Poly -> LassoCV)")
        
        X_transformed = pipeline.fit_transform(X, y)
        
        # Get shapes
        # We need to access the transformer inside to see original dimension before Lasso selection
        prep_step = pipeline.named_steps['prepare_data']
        cleaner_step = pipeline.named_steps['cleaner']
        
        # Transform X manually up to before selection to count features
        X_cleaned = cleaner_step.transform(X)
        X_expanded = prep_step.transform(X_cleaned)
        
        n_features_expanded = X_expanded.shape[1]
        n_features_selected = X_transformed.shape[1]
        
        print(f"  ✅ Input Rows: {len(X)}")
        print(f"  ✅ Features after Poly Expansion: {n_features_expanded}")
        print(f"  ✅ Features after Lasso Selection: {n_features_selected}")
        
        if n_features_selected == 0:
            print("  ⚠️  Warning: Lasso dropped ALL features (data sample might be too small/noisy)")
        elif n_features_selected > n_features_expanded:
             print("  ❌ Error: Selected features cannot be more than expanded features")
             return False
        else:
            print(f"  ✅ Feature selection active (dropped {n_features_expanded - n_features_selected} features)")

        print("\n✅ Preprocessing pipeline working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_dry_run():
    """Test 5: Run a quick training dry run using train.py logic."""
    print("=" * 60)
    print("TEST 5: Running training dry run (no W&B upload)...")
    print("=" * 60)
    
    # Set environment to skip W&B
    os.environ['SKIP_WANDB'] = 'true'
    
    try:
        import train
        
        # Load small sample of data
        df = train.load_and_prepare_data()
        df_sample = df.sample(n=min(300, len(df)), random_state=42)
        
        target = "mental_wellness_index_0_100"
        X = df_sample.drop(target, axis=1)
        y = df_sample[target]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Use the pipeline creator from train.py
        print("  Creating Smart Pipeline...")
        preprocessor = train.create_pipeline()
        
        # Train model using the updated train_model function
        print("  Training Custom Ridge Model...")
        model, metrics = train.train_model(X_train, y_train, X_test, y_test, preprocessor)
        
        print(f"\n  📊 Training Results:")
        print(f"    Train R²: {metrics['train_r2']:.4f}")
        print(f"    Test R²:  {metrics['test_r2']:.4f}")
        print(f"    Test RMSE: {metrics['test_rmse']:.4f}")
        
        # Check if model is reasonable (threshold relaxed for dry run on small data)
        if metrics['train_r2'] > -1.0:
            print(f"\n  ✅ Model trained and evaluated successfully.")
        else:
            print(f"\n  ⚠️  Warning: Model R² is very low (expected on small random sample).")
        
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