"""
MindSync Cluster Training Script with Weights & Biases Integration
K-Means Clustering (k=5) for Wellness Profiling

Produces one CSV per cluster with the average profile:
  dangerous_cluster_avg.csv
  not_healthy_cluster_avg.csv
  average_cluster_avg.csv
  above_average_cluster_avg.csv
  healthy_cluster_avg.csv

Cluster labels (ranked by mean mental_wellness_index, ascending):
  dangerous | not healthy | average | above average | healthy
"""

import os
import sys
import pandas as pd
import numpy as np
import wandb
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# CONFIGURATION
# ==========================================

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mindsync-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
ARTIFACTS_DIR = "artifacts"

N_CLUSTERS = 5
CLUSTER_LABELS_ORDERED = ["dangerous", "not healthy", "average", "above average", "healthy"]

NUMERICAL_COLS = [
    'age', 'work_screen_hours', 'leisure_screen_hours',
    'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
    'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week'
]
CATEGORICAL_COLS = ['gender', 'occupation', 'work_mode']
TARGET_COL = 'mental_wellness_index_0_100'

# Column order for output CSVs
OUTPUT_COLS = [
    'age', 'gender', 'occupation', 'work_mode',
    'work_screen_hours', 'leisure_screen_hours', 'sleep_hours',
    'sleep_quality_1_5', 'stress_level_0_10', 'productivity_0_100',
    'exercise_minutes_per_week', 'social_hours_per_week', 'mental_wellness_index_0_100'
]


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
    """Load and prepare dataset for clustering."""
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

    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)

    df = clean_occupation_column(df)
    return df


# ==========================================
# 2. CLUSTER PREPROCESSING
# ==========================================

def create_cluster_preprocessor():
    """
    Create preprocessing pipeline for K-Means clustering.
    StandardScaler for numerical + OneHotEncoder for categorical.
    """
    print("🔧 Creating cluster preprocessor (StandardScaler + OHE)...")

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop='first', handle_unknown='ignore', sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


# ==========================================
# 3. CLUSTERING & LABEL ASSIGNMENT
# ==========================================

def assign_cluster_labels(cluster_wellness_means):
    """
    Assign named labels to raw cluster IDs based on mean wellness score.
      Lowest wellness  → 'dangerous'
      Highest wellness → 'healthy'
    """
    sorted_clusters = cluster_wellness_means.sort_values().index.tolist()

    cluster_to_label = {
        cluster_id: label
        for cluster_id, label in zip(sorted_clusters, CLUSTER_LABELS_ORDERED)
    }

    print("\n📋 Cluster Label Mapping:")
    for cid in sorted(cluster_to_label.keys()):
        label = cluster_to_label[cid]
        wellness = cluster_wellness_means[cid]
        print(f"   Cluster {cid} → '{label}'  (mean wellness: {wellness:.2f})")

    return cluster_to_label


def train_clustering(df):
    """
    Fit K-Means (k=5) and return the labeled DataFrame and cluster→label mapping.
    """
    print(f"\n🎯 Training K-Means Clustering (k={N_CLUSTERS})...")

    X = df[NUMERICAL_COLS + CATEGORICAL_COLS].copy()

    preprocessor = create_cluster_preprocessor()
    X_scaled = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df = df.copy()
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    cluster_wellness_means = df.groupby('cluster_id')[TARGET_COL].mean()
    cluster_to_label = assign_cluster_labels(cluster_wellness_means)
    df['cluster_label'] = df['cluster_id'].map(cluster_to_label)

    print("\n📊 Cluster Summary:")
    for label in CLUSTER_LABELS_ORDERED:
        subset = df[df['cluster_label'] == label]
        print(
            f"   {label:<15} : {len(subset):>4} samples "
            f"| mean wellness: {subset[TARGET_COL].mean():.2f}"
        )

    return df, cluster_to_label


# ==========================================
# 4. CLUSTER AVERAGES
# ==========================================

def compute_cluster_averages(df):
    """
    Compute per-cluster profile:
      - Mean (rounded to 2 dp) for numerical columns
      - Mode for categorical columns
    Returns a DataFrame with one row per cluster label.
    """
    print("\n📐 Computing cluster averages...")
    rows = []

    for label in CLUSTER_LABELS_ORDERED:
        subset = df[df['cluster_label'] == label]
        row = {'cluster_label': label}

        for col in NUMERICAL_COLS:
            row[col] = round(float(subset[col].mean()), 2)

        for col in CATEGORICAL_COLS:
            row[col] = subset[col].mode().iloc[0]

        row['mental_wellness_index_0_100'] = round(float(subset[TARGET_COL].mean()), 2)

        rows.append(row)
        print(f"   ✅ {label}")

    return pd.DataFrame(rows)


# ==========================================
# 5. SAVE CSV ARTIFACTS & W&B
# ==========================================

def label_to_filename(label):
    """Convert cluster label to a safe filename, e.g. 'not healthy' → 'not_healthy_cluster_avg.csv'."""
    return label.replace(" ", "_") + "_cluster_avg.csv"


def save_artifacts_locally(cluster_avgs_df):
    """Save one CSV per cluster + a combined CSV."""
    print(f"\n💾 Saving CSV artifacts to {ARTIFACTS_DIR}/...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save per-cluster CSVs
    for _, row in cluster_avgs_df.iterrows():
        label = row['cluster_label']
        filename = label_to_filename(label)
        single_row = pd.DataFrame([row[OUTPUT_COLS]])
        single_row.to_csv(os.path.join(ARTIFACTS_DIR, filename), index=False)
        print(f"   ✅ {filename}")

    # Also save a combined all-clusters summary
    cluster_avgs_df.to_csv(
        os.path.join(ARTIFACTS_DIR, "cluster_averages.csv"), index=False
    )
    print(f"   ✅ cluster_averages.csv (all 5 clusters)")


def upload_to_wandb(df, cluster_avgs_df):
    """Upload clustering artifacts to Weights & Biases."""
    print("\n☁️ Uploading artifacts to Weights & Biases...")

    cluster_sizes = df['cluster_label'].value_counts().to_dict()
    cluster_wellness = df.groupby('cluster_label')[TARGET_COL].mean().to_dict()

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"kmeans-cluster-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model_type": "KMeans",
            "n_clusters": N_CLUSTERS,
            "cluster_labels": CLUSTER_LABELS_ORDERED,
            "numerical_features": NUMERICAL_COLS,
            "categorical_features": CATEGORICAL_COLS,
            "ranking_metric": TARGET_COL,
        },
        tags=["clustering", "kmeans", f"k={N_CLUSTERS}"],
    )

    wandb.log({
        **{f"wellness/{label}": cluster_wellness.get(label, 0) for label in CLUSTER_LABELS_ORDERED},
        **{f"size/{label}": cluster_sizes.get(label, 0) for label in CLUSTER_LABELS_ORDERED},
    })

    table = wandb.Table(dataframe=cluster_avgs_df)
    wandb.log({"cluster_averages_table": table})

    artifact = wandb.Artifact(
        name="mindsync-cluster-averages",
        type="dataset",
        description=f"MindSync K-Means Cluster Averages (k={N_CLUSTERS})",
        metadata={"cluster_labels": CLUSTER_LABELS_ORDERED},
    )

    # Upload every CSV
    for label in CLUSTER_LABELS_ORDERED:
        filepath = os.path.join(ARTIFACTS_DIR, label_to_filename(label))
        if os.path.exists(filepath):
            artifact.add_file(filepath)

    combined = os.path.join(ARTIFACTS_DIR, "cluster_averages.csv")
    if os.path.exists(combined):
        artifact.add_file(combined)

    run.log_artifact(artifact)
    print(f"✅ Artifacts uploaded to W&B. Run URL: {run.get_url()}")
    wandb.finish()


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    print("=" * 60)
    print("🚀 MindSync Cluster Training Pipeline (K-Means k=5)")
    print("=" * 60)

    # 1. Load Data
    df = load_and_prepare_data()

    # 2. Train K-Means Clustering
    df_labeled, cluster_to_label = train_clustering(df)

    # 3. Compute Cluster Averages
    cluster_avgs_df = compute_cluster_averages(df_labeled)

    # 4. Save CSVs Locally
    save_artifacts_locally(cluster_avgs_df)

    # 5. Upload to W&B
    if os.getenv("SKIP_WANDB") != "true":
        upload_to_wandb(df_labeled, cluster_avgs_df)
    else:
        print("⏭️ Skipping W&B upload (SKIP_WANDB=true)")

    print("\n" + "=" * 60)
    print("✅ Cluster Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
