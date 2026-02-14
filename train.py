"""
MindSync Model Training Script with Weights & Biases Integration
Updated with Smart Preprocessing (Yeo-Johnson + Poly + Lasso)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import custom ridge regression
# Pastikan file custom_ridge.py ada di folder yang sama
try:
    from custom_ridge import LinearRegressionRidge
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from custom_ridge import LinearRegressionRidge

# Configuration
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mindsync-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0-smart-lasso")
ARTIFACTS_DIR = "artifacts"


# ==========================================
# 1. DATA PREPARATION & CLEANING
# ==========================================

def clean_occupation_column(df):
    """Clean occupation column by combining rare categories."""
    df_copy = df.copy()
    if "occupation" in df_copy.columns:
        df_copy["occupation"] = df_copy["occupation"].replace(
            ["Unemployed", "Retired"], "Unemployed"
        )
    return df_copy


def load_and_prepare_data():
    """Load and prepare dataset for training."""
    print("📂 Loading dataset...")
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    
    df_path = os.path.join(base_dir, "df", "ScreenTime vs MentalWellness.csv")
    
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Dataset not found at {df_path}")

    df = pd.read_csv(df_path)
    print(f"✅ Loaded {len(df)} samples from {df_path}")
    
    # Drop user_id if exists
    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)
    
    return df


# ==========================================
# 2. SMART PIPELINE CONSTRUCTION
# ==========================================

def create_pipeline():
    """
    Create the Smart Preprocessing Pipeline:
    Cleaner -> Yeo-Johnson -> Poly -> OHE -> Global Scaling -> Lasso Selection
    """
    print("🔧 Creating smart preprocessing pipeline...")
    
    # Define Columns
    numerical_cols = [
        'age', 'work_screen_hours', 'leisure_screen_hours',
        'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
        'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
    ]
    categorical_cols = ['gender', 'occupation', 'work_mode']

    # 1. Custom Cleaner
    binning_transformer = FunctionTransformer(clean_occupation_column, validate=False)

    # 2. Numerical Transformer (Yeo-Johnson -> Poly)
    numerical_transformer = Pipeline([
        ('transform_skew', PowerTransformer(method='yeo-johnson')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
    ])

    # 3. Categorical Transformer
    categorical_transformer = OneHotEncoder(
        drop='first', handle_unknown='ignore', sparse_output=False
    )

    # 4. Column Transformer
    preprocessor_raw = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # 5. Lasso Engine for Feature Selection
    lasso_engine = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000)

    # 6. Full Pipeline Assembly
    # Note: We do NOT include the final model here yet, this is just the preprocessor
    smart_preprocessor = Pipeline([
        ('cleaner', binning_transformer),
        ('prepare_data', preprocessor_raw), 
        ('global_scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(lasso_engine)) 
    ])
    
    return smart_preprocessor


# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================

def train_model(X_train, y_train, X_test, y_test, preprocessor):
    """Train the Pipeline with Custom Ridge."""
    print("🎯 Training model (Lasso Selection + Custom Ridge)...")
    
    # Full pipeline with model
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegressionRidge(alpha=1.0, solver="closed_form")),
    ])
    
    # Train
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)
    
    metrics = {
        "train_r2": r2_score(y_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"✅ Training R²: {metrics['train_r2']:.4f}")
    print(f"✅ Test R²: {metrics['test_r2']:.4f}")
    print(f"✅ Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model_pipeline, metrics


def extract_feature_importance(model_pipeline):
    """
    Extract feature importance handling Polynomial expansion and Lasso selection.
    """
    print("📊 Extracting advanced feature importance...")
    
    try:
        # 1. Dig into the pipeline steps
        preprocessor_pipe = model_pipeline.named_steps['preprocessor']
        # Support both step names: 'model' (notebook convention) and 'regressor' (legacy)
        regressor = model_pipeline.named_steps.get('model') or model_pipeline.named_steps.get('regressor')
        
        # Steps inside preprocessor
        prepare_step = preprocessor_pipe.named_steps['prepare_data']
        selector_step = preprocessor_pipe.named_steps['feature_selection']
        
        # 2. Get ALL feature names (after Poly + OHE)
        all_feature_names = prepare_step.get_feature_names_out()
        
        # 3. Get Mask from Lasso Selection
        selected_mask = selector_step.get_support()
        
        # 4. Filter names
        final_feature_names = all_feature_names[selected_mask]
        
        # 5. Get Coefficients from Ridge (try 'model' step first, fallback to 'regressor')
        coefficients = regressor.coef_
        
        # Validate shapes
        if len(final_feature_names) != len(coefficients):
            print(f"⚠️ Warning: Shape mismatch. Names: {len(final_feature_names)}, Coefs: {len(coefficients)}")
            return pd.DataFrame() # Return empty if mismatch
            
        # 6. Create DataFrame
        # Use uppercase column names to match Flask's analyze_wellness_factors()
        # which reads row["Feature"] and row["Coefficient"]
        coef_df = pd.DataFrame({
            "Feature": final_feature_names, 
            "Coefficient": coefficients, 
            "Abs_Coefficient": np.abs(coefficients)
        })
        coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)
        
        print(f"✅ Extracted {len(coef_df)} selected features (from original {len(all_feature_names)})")
        return coef_df
        
    except Exception as e:
        print(f"⚠️ Failed to extract feature importance: {e}")
        return pd.DataFrame()


# ==========================================
# 4. ARTIFACT HANDLING & W&B
# ==========================================

def save_artifacts_locally(model_pipeline, coef_df):
    """Save artifacts locally."""
    print(f"💾 Saving artifacts to {ARTIFACTS_DIR}/...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Save Full Pipeline (Model + Preprocessor)
    # This is the only file needed for inference
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model_pipeline, f)
    
    # Also save just the preprocessor (optional, useful for debugging)
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(model_pipeline.named_steps['preprocessor'], f)
    
    # Save coefficients as both files:
    # - model_coefficients.csv: Used by Flask's analyze_wellness_factors()
    # - feature_importance.csv: Used by W&B and general analysis
    if not coef_df.empty:
        coef_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_coefficients.csv"), index=False)
        coef_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
    
    print("✅ Artifacts saved locally")


def upload_to_wandb(metrics, model_config, coef_df):
    """Upload artifacts to Weights & Biases."""
    print("☁️ Uploading artifacts to Weights & Biases...")
    
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"smart-train-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config=model_config,
        tags=[MODEL_VERSION, "smart-preprocessing", "lasso-selection", "poly-features"],
    )
    
    # Log metrics
    wandb.log(metrics)
    
    # Log Feature Importance Plot
    if not coef_df.empty:
        top_features = coef_df.head(20)
        table = wandb.Table(dataframe=top_features)
        wandb.log({"feature_importance_plot": wandb.plot.bar(
            table, "feature", "coefficient", title="Top 20 Features (Coefficients)"
        )})
    
    # Create artifact
    artifact = wandb.Artifact(
        name="mindsync-model-smart",
        type="model",
        description="MindSync Model with Poly+Lasso Preprocessing",
        metadata=metrics,
    )
    
    # Add files
    files = ["model.pkl", "preprocessor.pkl", "feature_importance.csv", "model_coefficients.csv"]
    for filename in files:
        filepath = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(filepath):
            artifact.add_file(filepath)
    
    run.log_artifact(artifact)
    print(f"✅ Artifacts uploaded to W&B. Run URL: {run.get_url()}")
    wandb.finish()


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    print("=" * 60)
    print("🚀 MindSync Training Pipeline (Smart V2)")
    print("=" * 60)
    
    # 1. Load Data
    df = load_and_prepare_data()
    target = "mental_wellness_index_0_100"
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Create Smart Preprocessor
    preprocessor = create_pipeline()
    
    # 3. Train Model
    model_pipeline, metrics = train_model(X_train, y_train, X_test, y_test, preprocessor)
    
    # 4. Extract Importance
    coef_df = extract_feature_importance(model_pipeline)
    
    # 5. Save Locally
    save_artifacts_locally(model_pipeline, coef_df)
    
    # 6. Upload to W&B
    if os.getenv("SKIP_WANDB") != "true":
        # Get Lasso Alpha if available for config logging
        try:
            lasso_alpha = model_pipeline.named_steps['preprocessor'].named_steps['feature_selection'].estimator_.alpha_
        except:
            lasso_alpha = "unknown"

        model_config = {
            "model_type": "LinearRegressionRidge",
            "preprocessing": "Yeo-Johnson + Poly(deg=2) + Lasso",
            "lasso_alpha_selected": lasso_alpha,
            "n_features_input": X_train.shape[1],
            "n_features_selected": len(coef_df)
        }
        
        upload_to_wandb(metrics, model_config, coef_df)
    else:
        print("⏭️ Skipping W&B upload (SKIP_WANDB=true)")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()