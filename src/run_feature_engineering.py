"""
Feature Engineering Example Script

This script demonstrates how to use the comprehensive feature engineering pipeline
for the credit risk model. It includes all the required transformations:

1. Aggregate Features (transaction counts, amounts, etc.)
2. Temporal Features (hour, day, month, etc.)
3. Categorical Encoding (One-Hot, Label, WOE)
4. Missing Value Handling
5. Feature Scaling and Normalization
6. Advanced Techniques (Weight of Evidence, Information Value)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processing import (
    process_data_pipeline,
    load_data,
    validate_data,
    identify_feature_types,
    get_feature_summary
)
from feature_engineering import (
    create_feature_engineering_pipeline,
    create_advanced_feature_engineering_pipeline,
    get_feature_importance_info
)


def main():
    """Main function to run the feature engineering pipeline."""
    
    print("=" * 80)
    print("🚀 CREDIT RISK MODEL - FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # Configuration
    data_path = "../data/raw/data.csv"
    output_path = "../data/processed/processed_data.csv"
    
    # Create output directory if it doesn't exist
    Path("../data/processed").mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load and explore raw data
        print("\n📊 STEP 1: Loading and exploring raw data...")
        df_raw = load_data(data_path)
        
        # Basic data exploration
        print(f"\n📈 Raw data shape: {df_raw.shape}")
        print(f"📊 Data types:")
        print(df_raw.dtypes.value_counts())
        
        # Validation report
        validation_report = validate_data(df_raw)
        print(f"\n🔍 Validation report:")
        print(f"   Missing values: {sum(validation_report['missing_values'].values())}")
        print(f"   Duplicates: {validation_report['duplicates']}")
        print(f"   Quality issues: {len(validation_report['quality_issues'])}")
        
        # Feature types identification
        feature_types = identify_feature_types(df_raw)
        print(f"\n🏷️  Feature types:")
        for feature_type, columns in feature_types.items():
            print(f"   {feature_type}: {len(columns)} features")
            if columns:
                print(f"      Examples: {columns[:3]}")
        
        # Step 2: Run standard feature engineering pipeline
        print("\n" + "=" * 60)
        print("🔧 STEP 2: Running standard feature engineering pipeline...")
        print("=" * 60)
        
        X_processed_std, y_std, processing_info_std = process_data_pipeline(
            data_path=data_path,
            output_path=output_path.replace('.csv', '_standard.csv'),
            pipeline_type='standard',
            encoding_strategy='label',
            scaling_method='standard',
            feature_selection_k=50
        )
        
        print(f"\n✅ Standard pipeline results:")
        print(f"   Original features: {processing_info_std['original_shape'][1]}")
        print(f"   Processed features: {processing_info_std['processed_shape'][1]}")
        print(f"   Top features: {processing_info_std['feature_importance']['top_features'][:5]}")
        
        # Step 3: Run advanced feature engineering pipeline
        print("\n" + "=" * 60)
        print("🚀 STEP 3: Running advanced feature engineering pipeline...")
        print("=" * 60)
        
        X_processed_adv, y_adv, processing_info_adv = process_data_pipeline(
            data_path=data_path,
            output_path=output_path.replace('.csv', '_advanced.csv'),
            pipeline_type='advanced',
            encoding_strategy='woe',
            scaling_method='standard',
            feature_selection_k=50
        )
        
        print(f"\n✅ Advanced pipeline results:")
        print(f"   Original features: {processing_info_adv['original_shape'][1]}")
        print(f"   Processed features: {processing_info_adv['processed_shape'][1]}")
        print(f"   Top features: {processing_info_adv['feature_importance']['top_features'][:5]}")
        
        # Step 4: Compare pipelines
        print("\n" + "=" * 60)
        print("📊 STEP 4: Pipeline comparison and analysis...")
        print("=" * 60)
        
        # Compare feature importance
        std_correlations = processing_info_std['feature_importance']['correlations']
        adv_correlations = processing_info_adv['feature_importance']['correlations']
        
        print(f"\n🔍 Feature importance comparison:")
        print(f"   Standard pipeline - Top 5 features:")
        for i, (feature, corr) in enumerate(list(std_correlations.items())[:5]):
            print(f"      {i+1}. {feature}: {corr:.4f}")
        
        print(f"\n   Advanced pipeline - Top 5 features:")
        for i, (feature, corr) in enumerate(list(adv_correlations.items())[:5]):
            print(f"      {i+1}. {feature}: {corr:.4f}")
        
        # Step 5: Create visualizations
        print("\n" + "=" * 60)
        print("📈 STEP 5: Creating visualizations...")
        print("=" * 60)
        
        create_feature_engineering_visualizations(
            processing_info_std, processing_info_adv, output_path
        )
        
        # Step 6: Generate comprehensive report
        print("\n" + "=" * 60)
        print("📋 STEP 6: Generating comprehensive report...")
        print("=" * 60)
        
        generate_feature_engineering_report(
            df_raw, processing_info_std, processing_info_adv, output_path
        )
        
        print("\n" + "=" * 80)
        print("🎉 FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Processed data saved to: {output_path}")
        print(f"📊 Report saved to: {output_path.replace('.csv', '_report.txt')}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in feature engineering pipeline: {e}")
        raise


def create_feature_engineering_visualizations(processing_info_std, processing_info_adv, output_path):
    """Create visualizations for feature engineering results."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Engineering Pipeline Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature importance comparison (Standard)
    std_correlations = processing_info_std['feature_importance']['correlations']
    top_features_std = list(std_correlations.items())[:10]
    features_std, correlations_std = zip(*top_features_std)
    
    axes[0, 0].barh(range(len(features_std)), correlations_std, color='skyblue')
    axes[0, 0].set_yticks(range(len(features_std)))
    axes[0, 0].set_yticklabels(features_std, fontsize=8)
    axes[0, 0].set_title('Standard Pipeline - Top 10 Features', fontweight='bold')
    axes[0, 0].set_xlabel('Correlation with Target')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature importance comparison (Advanced)
    adv_correlations = processing_info_adv['feature_importance']['correlations']
    top_features_adv = list(adv_correlations.items())[:10]
    features_adv, correlations_adv = zip(*top_features_adv)
    
    axes[0, 1].barh(range(len(features_adv)), correlations_adv, color='lightcoral')
    axes[0, 1].set_yticks(range(len(features_adv)))
    axes[0, 1].set_yticklabels(features_adv, fontsize=8)
    axes[0, 1].set_title('Advanced Pipeline - Top 10 Features', fontweight='bold')
    axes[0, 1].set_xlabel('Correlation with Target')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature count comparison
    pipeline_names = ['Standard', 'Advanced']
    feature_counts = [
        processing_info_std['processed_shape'][1],
        processing_info_adv['processed_shape'][1]
    ]
    
    axes[1, 0].bar(pipeline_names, feature_counts, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_title('Number of Features Generated', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(feature_counts):
        axes[1, 0].text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Feature type distribution
    feature_types_std = processing_info_std['feature_types']
    feature_types_adv = processing_info_adv['feature_types']
    
    # Count feature types
    std_counts = {k: len(v) for k, v in feature_types_std.items() if v}
    adv_counts = {k: len(v) for k, v in feature_types_adv.items() if v}
    
    # Create comparison
    all_types = set(std_counts.keys()) | set(adv_counts.keys())
    std_values = [std_counts.get(t, 0) for t in all_types]
    adv_values = [adv_counts.get(t, 0) for t in all_types]
    
    x = np.arange(len(all_types))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, std_values, width, label='Standard', color='skyblue', alpha=0.8)
    axes[1, 1].bar(x + width/2, adv_values, width, label='Advanced', color='lightcoral', alpha=0.8)
    axes[1, 1].set_title('Feature Type Distribution', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(all_types, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path.replace('.csv', '_visualizations.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved to: {plot_path}")
    plt.show()


def generate_feature_engineering_report(df_raw, processing_info_std, processing_info_adv, output_path):
    """Generate a comprehensive feature engineering report."""
    
    report_path = output_path.replace('.csv', '_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CREDIT RISK MODEL - FEATURE ENGINEERING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Raw data summary
        f.write("1. RAW DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset shape: {df_raw.shape}\n")
        f.write(f"Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"Missing values: {df_raw.isnull().sum().sum()}\n")
        f.write(f"Duplicate rows: {df_raw.duplicated().sum()}\n\n")
        
        # Feature types
        feature_types = identify_feature_types(df_raw)
        f.write("Feature types:\n")
        for feature_type, columns in feature_types.items():
            f.write(f"  {feature_type}: {len(columns)} features\n")
        f.write("\n")
        
        # Standard pipeline results
        f.write("2. STANDARD PIPELINE RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original features: {processing_info_std['original_shape'][1]}\n")
        f.write(f"Processed features: {processing_info_std['processed_shape'][1]}\n")
        f.write(f"Encoding strategy: {processing_info_std['encoding_strategy']}\n")
        f.write(f"Scaling method: {processing_info_std['scaling_method']}\n\n")
        
        f.write("Top 10 features by correlation:\n")
        std_correlations = processing_info_std['feature_importance']['correlations']
        for i, (feature, corr) in enumerate(list(std_correlations.items())[:10]):
            f.write(f"  {i+1:2d}. {feature}: {corr:.4f}\n")
        f.write("\n")
        
        # Advanced pipeline results
        f.write("3. ADVANCED PIPELINE RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original features: {processing_info_adv['original_shape'][1]}\n")
        f.write(f"Processed features: {processing_info_adv['processed_shape'][1]}\n")
        f.write(f"Encoding strategy: {processing_info_adv['encoding_strategy']}\n")
        f.write(f"Scaling method: {processing_info_adv['scaling_method']}\n\n")
        
        f.write("Top 10 features by correlation:\n")
        adv_correlations = processing_info_adv['feature_importance']['correlations']
        for i, (feature, corr) in enumerate(list(adv_correlations.items())[:10]):
            f.write(f"  {i+1:2d}. {feature}: {corr:.4f}\n")
        f.write("\n")
        
        # Pipeline comparison
        f.write("4. PIPELINE COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"Feature count difference: {processing_info_adv['processed_shape'][1] - processing_info_std['processed_shape'][1]}\n")
        
        # Find common top features
        std_top = set(processing_info_std['feature_importance']['top_features'][:10])
        adv_top = set(processing_info_adv['feature_importance']['top_features'][:10])
        common_features = std_top & adv_top
        
        f.write(f"Common top features: {len(common_features)}\n")
        for feature in sorted(common_features):
            f.write(f"  - {feature}\n")
        f.write("\n")
        
        # Recommendations
        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("Based on the analysis:\n")
        f.write("1. Use the advanced pipeline for better feature representation\n")
        f.write("2. Focus on temporal and customer aggregate features\n")
        f.write("3. Consider WOE encoding for categorical variables\n")
        f.write("4. Monitor feature importance for model interpretability\n")
        f.write("5. Validate feature stability across different time periods\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main() 