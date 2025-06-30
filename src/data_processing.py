"""
Data Processing Module for Credit Risk Model

This module provides comprehensive data processing functionality including:
- Data loading and validation
- Feature engineering using sklearn pipelines
- Data transformation and preprocessing
- Model-ready data preparation
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineering module
from .feature_engineering import (
    create_feature_engineering_pipeline,
    create_advanced_feature_engineering_pipeline,
    get_feature_importance_info,
    save_feature_engineering_pipeline,
    load_feature_engineering_pipeline
)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file with basic validation.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def validate_data(df: pd.DataFrame) -> Dict:
    """
    Validate the dataset and return validation report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'duplicates': df.duplicated().sum(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    
    # Check for required columns
    required_cols = ['TransactionId', 'CustomerId', 'Amount', 'TransactionStartTime', 'FraudResult']
    missing_cols = [col for col in required_cols if col not in df.columns]
    validation_report['missing_required_cols'] = missing_cols
    
    # Check data quality
    validation_report['quality_issues'] = []
    
    if df.isnull().sum().sum() > 0:
        validation_report['quality_issues'].append("Missing values detected")
    
    if df.duplicated().sum() > 0:
        validation_report['quality_issues'].append("Duplicate rows detected")
    
    if len(missing_cols) > 0:
        validation_report['quality_issues'].append(f"Missing required columns: {missing_cols}")
    
    return validation_report


def identify_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify different types of features in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with feature types
    """
    feature_types = {
        'categorical': [],
        'numerical': [],
        'datetime': [],
        'id_columns': [],
        'target': []
    }
    
    for col in df.columns:
        if col.lower() in ['fraudresult', 'target', 'label']:
            feature_types['target'].append(col)
        elif col.lower().endswith('id') or col in ['TransactionId', 'CustomerId', 'AccountId', 'SubscriptionId']:
            feature_types['id_columns'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'time' in col.lower() or 'date' in col.lower():
            feature_types['datetime'].append(col)
        elif df[col].dtype in ['object', 'category', 'string']:
            feature_types['categorical'].append(col)
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_types['numerical'].append(col)
    
    return feature_types


def compute_rfm(df: pd.DataFrame, customer_id_col: str, date_col: str, amount_col: str, ref_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary features per customer.
    
    Args:
        df: DataFrame with transactions
        customer_id_col: column name for customer ID
        date_col: column name for transaction date
        amount_col: column name for transaction amount
        ref_date: reference date for recency calculation (default: max date in df)
        
    Returns:
        DataFrame with columns: CustomerId, recency_days, frequency, monetary
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if ref_date is None:
        ref_date = df[date_col].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby(customer_id_col).agg(
        recency_days=(date_col, lambda x: (ref_date - x.max()).days),
        frequency=(date_col, 'count'),
        monetary=(amount_col, 'sum')
    ).reset_index()
    
    return rfm


def create_customer_features(df: pd.DataFrame, customer_col: str = 'CustomerId') -> pd.DataFrame:
    """
    Create comprehensive customer-level features.
    
    Args:
        df: Input DataFrame
        customer_col: Customer ID column name
        
    Returns:
        DataFrame with customer features
    """
    customer_features = df.groupby(customer_col).agg({
        'Amount': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
        'Value': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
        'TransactionStartTime': ['min', 'max', 'count']
    }).fillna(0)
    
    # Flatten column names
    customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
    
    # Calculate additional features
    customer_features['customer_lifetime_days'] = (
        pd.to_datetime(customer_features['TransactionStartTime_max']) - 
        pd.to_datetime(customer_features['TransactionStartTime_min'])
    ).dt.days
    
    customer_features['customer_avg_transaction_value'] = (
        customer_features['Value_sum'] / customer_features['Value_count']
    )
    
    customer_features['customer_transaction_frequency'] = (
        customer_features['Amount_count'] / (customer_features['customer_lifetime_days'] + 1)
    )
    
    return customer_features.reset_index()


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-level features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with transaction features
    """
    df = df.copy()
    
    # Convert datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract temporal features
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    df['transaction_dayofyear'] = df['TransactionStartTime'].dt.dayofyear
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['transaction_day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['transaction_day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)
    
    # Business indicators
    df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
    df['is_weekend'] = (df['transaction_dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    
    # Amount-based features
    df['amount_log'] = np.log1p(np.abs(df['Amount']))
    df['value_log'] = np.log1p(np.abs(df['Value']))
    df['amount_value_ratio'] = df['Amount'] / (df['Value'] + 1e-8)
    
    return df


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = 'FraudResult',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling with train-test split.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test


def process_data_pipeline(
    data_path: str,
    output_path: Optional[str] = None,
    pipeline_type: str = 'standard',
    encoding_strategy: str = 'label',
    scaling_method: str = 'standard',
    feature_selection_k: int = 50
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Complete data processing pipeline from raw data to model-ready format.
    
    Args:
        data_path: Path to raw data file
        output_path: Path to save processed data (optional)
        pipeline_type: Type of pipeline ('standard' or 'advanced')
        encoding_strategy: Categorical encoding strategy
        scaling_method: Feature scaling method
        feature_selection_k: Number of features to select
        
    Returns:
        Tuple of (X_processed, y, processing_info)
    """
    print("🚀 Starting data processing pipeline...")
    
    # Step 1: Load and validate data
    print("📊 Loading and validating data...")
    df = load_data(data_path)
    validation_report = validate_data(df)
    
    if validation_report['quality_issues']:
        print("⚠️  Quality issues detected:")
        for issue in validation_report['quality_issues']:
            print(f"   - {issue}")
    
    # Step 2: Identify feature types
    print("🔍 Identifying feature types...")
    feature_types = identify_feature_types(df)
    
    print(f"   Categorical features: {len(feature_types['categorical'])}")
    print(f"   Numerical features: {len(feature_types['numerical'])}")
    print(f"   Datetime features: {len(feature_types['datetime'])}")
    print(f"   ID columns: {len(feature_types['id_columns'])}")
    
    # Step 3: Create basic features
    print("🔧 Creating basic features...")
    df = create_transaction_features(df)
    
    # Step 4: Create customer features
    print("👥 Creating customer features...")
    customer_features = create_customer_features(df)
    df = df.merge(customer_features, on='CustomerId', how='left')
    
    # Step 5: Prepare for modeling
    print("🎯 Preparing data for modeling...")
    X_train, y_train, X_test, y_test = prepare_model_data(df)
    
    # Step 6: Create and fit feature engineering pipeline
    print("⚙️  Creating feature engineering pipeline...")
    
    if pipeline_type == 'advanced':
        pipeline = create_advanced_feature_engineering_pipeline(
            categorical_cols=feature_types['categorical'],
            numerical_cols=feature_types['numerical'],
            datetime_col='TransactionStartTime',
            customer_col='CustomerId',
            amount_col='Amount',
            target_col='FraudResult'
        )
    else:
        pipeline = create_feature_engineering_pipeline(
            categorical_cols=feature_types['categorical'],
            numerical_cols=feature_types['numerical'],
            datetime_col='TransactionStartTime',
            customer_col='CustomerId',
            amount_col='Amount',
            target_col='FraudResult',
            encoding_strategy=encoding_strategy,
            scaling_method=scaling_method,
            feature_selection_k=feature_selection_k
        )
    
    # Fit pipeline on training data
    print("🔧 Fitting feature engineering pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Transform data
    print("🔄 Transforming data...")
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Get feature importance information
    feature_info = get_feature_importance_info(pipeline, X_train, y_train)
    
    # Combine train and test for final output
    X_processed = pd.concat([X_train_processed, X_test_processed], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    
    # Save pipeline if output path provided
    if output_path:
        pipeline_path = output_path.replace('.csv', '_pipeline.pkl')
        save_feature_engineering_pipeline(pipeline, pipeline_path)
        print(f"💾 Pipeline saved to: {pipeline_path}")
    
    # Save processed data if output path provided
    if output_path:
        processed_data = X_processed.copy()
        processed_data[target_col] = y_combined
        processed_data.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
    
    # Compile processing information
    processing_info = {
        'original_shape': df.shape,
        'processed_shape': X_processed.shape,
        'feature_types': feature_types,
        'validation_report': validation_report,
        'feature_importance': feature_info,
        'pipeline_type': pipeline_type,
        'encoding_strategy': encoding_strategy,
        'scaling_method': scaling_method
    }
    
    print("✅ Data processing pipeline completed successfully!")
    print(f"   Original features: {df.shape[1]}")
    print(f"   Processed features: {X_processed.shape[1]}")
    print(f"   Top features: {feature_info['top_features'][:5]}")
    
    return X_processed, y_combined, processing_info


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive feature summary.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with feature summary
    """
    summary = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': df.isnull().sum().to_dict()
        },
        'data_types': df.dtypes.value_counts().to_dict(),
        'numerical_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        'categorical_summary': {}
    }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    return summary 