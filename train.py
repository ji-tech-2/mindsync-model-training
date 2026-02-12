"""
MindSync Model Training Script with Weights & Biases Integration

This script trains the mental wellness prediction model and uploads
artifacts to Weights & Biases for use by the inference service.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans

# Import custom ridge regression
from custom_ridge import LinearRegressionRidge

# Configuration
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mindsync-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)  # Your W&B username/team
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
ARTIFACTS_DIR = "artifacts"


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
    
    # Determine base directory
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    
    df_path = os.path.join(base_dir, "df", "ScreenTime vs MentalWellness.csv")
    df = pd.read_csv(df_path)
    print(f"✅ Loaded {len(df)} samples from {df_path}")
    
    # Clean occupation
    df = clean_occupation_column(df)
    
    return df


def create_preprocessor(categorical_features, numeric_features):
    """Create preprocessing pipeline."""
    print("🔧 Creating preprocessing pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )
    
    return preprocessor


def train_model(X_train, y_train, X_test, y_test, preprocessor):
    """Train the Ridge Regression model."""
    print("🎯 Training model...")
    
    # Create pipeline
    model_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegressionRidge(alpha=1.0, solver="closed_form")),
        ]
    )
    
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


def perform_clustering(X_processed, y, n_clusters=3):
    """Perform K-Means clustering on processed features."""
    print(f"🔍 Performing K-Means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_processed)
    
    # Create dataframe with clusters
    df_clustered = pd.DataFrame(X_processed)
    df_clustered["cluster"] = clusters
    df_clustered["mental_wellness_score"] = y.values
    
    # Find healthy cluster (highest average mental wellness score)
    cluster_means = df_clustered.groupby("cluster")["mental_wellness_score"].mean()
    healthy_cluster = cluster_means.idxmax()
    
    print(f"✅ Healthy cluster: {healthy_cluster}")
    
    # Get average values for healthy cluster
    healthy_cluster_data = df_clustered[df_clustered["cluster"] == healthy_cluster]
    healthy_cluster_avg = healthy_cluster_data.drop(
        ["cluster", "mental_wellness_score"], axis=1
    ).mean()
    
    return healthy_cluster_avg


def extract_feature_importance(model_pipeline):
    """Extract feature importance from the trained model."""
    print("📊 Extracting feature importance...")
    
    # Get the trained model
    regressor = model_pipeline.named_steps["regressor"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    
    # Get feature names after preprocessing
    numeric_features = preprocessor.named_transformers_["num"].get_feature_names_out()
    categorical_features = preprocessor.named_transformers_["cat"].get_feature_names_out()
    feature_names = list(numeric_features) + list(categorical_features)
    
    # Get coefficients
    coefficients = regressor.coef_
    
    # Create dataframe
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients, "abs_coefficient": np.abs(coefficients)}
    )
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    
    print(f"✅ Extracted {len(coef_df)} feature importances")
    
    return coef_df


def save_artifacts_locally(model, preprocessor, healthy_cluster_avg, coef_df):
    """Save artifacts locally before uploading to W&B."""
    print(f"💾 Saving artifacts to {ARTIFACTS_DIR}/...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Save model
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Save preprocessor
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    
    # Save healthy cluster averages
    healthy_cluster_avg.to_csv(
        os.path.join(ARTIFACTS_DIR, "healthy_cluster_avg.csv"), index=False
    )
    
    # Save model coefficients
    coef_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_coefficients.csv"), index=False)
    
    # Save feature importance (top 20)
    top_features = coef_df.head(20)
    top_features.to_csv(
        os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False
    )
    
    print("✅ Artifacts saved locally")


def upload_to_wandb(metrics, model_config):
    """Upload artifacts and metadata to Weights & Biases."""
    print("☁️ Uploading artifacts to Weights & Biases...")
    
    # Initialize W&B run
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=model_config,
        tags=[MODEL_VERSION, "ridge-regression"],
    )
    
    # Log metrics
    wandb.log(metrics)
    
    # Create artifact
    artifact = wandb.Artifact(
        name="mindsync-model",
        type="model",
        description="MindSync mental wellness prediction model with preprocessing pipeline",
        metadata={
            "model_version": MODEL_VERSION,
            "training_date": datetime.now().isoformat(),
            "model_type": "LinearRegressionRidge",
            **metrics,
        },
    )
    
    # Add all files from artifacts directory
    artifact.add_dir(ARTIFACTS_DIR)
    
    # Log artifact
    run.log_artifact(artifact)
    
    print(f"✅ Artifacts uploaded to W&B project: {WANDB_PROJECT}")
    print(f"📊 View your run at: {run.get_url()}")
    
    # Finish run
    wandb.finish()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🚀 MindSync Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Define features
    target = "mental_wellness_score"
    categorical_features = ["gender", "occupation", "platform"]
    numeric_features = [
        col
        for col in df.columns
        if col not in categorical_features + [target, "user_id"]
    ]
    
    print(f"📊 Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Split data
    X = df[numeric_features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(categorical_features, numeric_features)
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_test, y_test, preprocessor)
    
    # Perform clustering on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    healthy_cluster_avg = perform_clustering(X_train_processed, y_train)
    
    # Convert to DataFrame for saving
    healthy_cluster_df = pd.DataFrame([healthy_cluster_avg])
    
    # Extract feature importance
    coef_df = extract_feature_importance(model)
    
    # Save artifacts locally
    save_artifacts_locally(model, preprocessor, healthy_cluster_df, coef_df)
    
    # Model configuration
    model_config = {
        "model_type": "LinearRegressionRidge",
        "alpha": 1.0,
        "solver": "closed_form",
        "n_features": len(numeric_features) + len(categorical_features),
        "n_numeric": len(numeric_features),
        "n_categorical": len(categorical_features),
    }
    
    # Upload to W&B
    if os.getenv("SKIP_WANDB") != "true":
        upload_to_wandb(metrics, model_config)
    else:
        print("⏭️ Skipping W&B upload (SKIP_WANDB=true)")
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
