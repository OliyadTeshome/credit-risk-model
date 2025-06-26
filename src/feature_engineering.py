"""
Feature Engineering Module for Credit Risk Model

This module provides a comprehensive feature engineering pipeline that includes:
- Aggregate features (transaction counts, amounts, etc.)
- Temporal features (hour, day, month, etc.)
- Categorical encoding (One-Hot, Label, WOE)
- Missing value handling
- Feature scaling and normalization
- Advanced techniques (Weight of Evidence, Information Value)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer

# Advanced feature engineering
try:
    from xverse.transformer import MonotonicBinning
    from woe import WOE
    from category_encoders import TargetEncoder, WOEEncoder
    XVERSE_AVAILABLE = True
    WOE_AVAILABLE = True
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False
    WOE_AVAILABLE = False
    CATEGORY_ENCODERS_AVAILABLE = False
    print("Warning: xverse, woe, or category_encoders not available. Some advanced features will be disabled.")


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""
    
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        self.datetime_col = datetime_col
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        # Extract temporal features
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        X['transaction_dayofweek'] = X[self.datetime_col].dt.dayofweek
        X['transaction_dayofyear'] = X[self.datetime_col].dt.dayofyear
        X['transaction_week'] = X[self.datetime_col].dt.isocalendar().week
        X['transaction_quarter'] = X[self.datetime_col].dt.quarter
        
        # Cyclical encoding for periodic features
        X['hour_sin'] = np.sin(2 * np.pi * X['transaction_hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['transaction_hour'] / 24)
        X['day_sin'] = np.sin(2 * np.pi * X['transaction_day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['transaction_day'] / 31)
        X['month_sin'] = np.sin(2 * np.pi * X['transaction_month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['transaction_month'] / 12)
        X['dayofweek_sin'] = np.sin(2 * np.pi * X['transaction_dayofweek'] / 7)
        X['dayofweek_cos'] = np.cos(2 * np.pi * X['transaction_dayofweek'] / 7)
        
        # Business hours indicator
        X['is_business_hours'] = ((X['transaction_hour'] >= 9) & (X['transaction_hour'] <= 17)).astype(int)
        X['is_weekend'] = (X['transaction_dayofweek'] >= 5).astype(int)
        X['is_night'] = ((X['transaction_hour'] >= 22) | (X['transaction_hour'] <= 6)).astype(int)
        
        # Remove original datetime column
        X = X.drop(columns=[self.datetime_col])
        
        return X


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create aggregate features per customer."""
    
    def __init__(self, customer_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.customer_stats_ = None
        
    def fit(self, X, y=None):
        # Calculate customer statistics
        self.customer_stats_ = X.groupby(self.customer_col)[self.amount_col].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max', 'median'
        ]).fillna(0)
        
        # Rename columns
        self.customer_stats_.columns = [
            f'customer_{self.amount_col}_count',
            f'customer_{self.amount_col}_total',
            f'customer_{self.amount_col}_mean',
            f'customer_{self.amount_col}_std',
            f'customer_{self.amount_col}_min',
            f'customer_{self.amount_col}_max',
            f'customer_{self.amount_col}_median'
        ]
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Merge customer statistics
        X = X.merge(self.customer_stats_, left_on=self.customer_col, right_index=True, how='left')
        
        # Fill missing values for new customers
        agg_cols = [col for col in X.columns if col.startswith(f'customer_{self.amount_col}_')]
        X[agg_cols] = X[agg_cols].fillna(0)
        
        # Create ratio features
        X[f'{self.amount_col}_to_customer_mean'] = X[self.amount_col] / (X[f'customer_{self.amount_col}_mean'] + 1e-8)
        X[f'{self.amount_col}_to_customer_std'] = (X[self.amount_col] - X[f'customer_{self.amount_col}_mean']) / (X[f'customer_{self.amount_col}_std'] + 1e-8)
        
        return X


class CategoricalFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features using multiple strategies."""
    
    def __init__(self, categorical_cols: List[str], encoding_strategy: str = 'onehot'):
        self.categorical_cols = categorical_cols
        self.encoding_strategy = encoding_strategy
        self.encoders_ = {}
        self.label_encoders_ = {}
        
    def fit(self, X, y=None):
        for col in self.categorical_cols:
            if col in X.columns:
                if self.encoding_strategy == 'label':
                    self.label_encoders_[col] = LabelEncoder()
                    self.label_encoders_[col].fit(X[col].astype(str))
                elif self.encoding_strategy == 'target' and y is not None and CATEGORY_ENCODERS_AVAILABLE:
                    self.encoders_[col] = TargetEncoder()
                    self.encoders_[col].fit(X[col], y)
                elif self.encoding_strategy == 'woe' and y is not None and CATEGORY_ENCODERS_AVAILABLE:
                    self.encoders_[col] = WOEEncoder()
                    self.encoders_[col].fit(X[col], y)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.categorical_cols:
            if col in X.columns:
                if self.encoding_strategy == 'label' and col in self.label_encoders_:
                    X[col] = self.label_encoders_[col].transform(X[col].astype(str))
                elif self.encoding_strategy in ['target', 'woe'] and col in self.encoders_:
                    encoded = self.encoders_[col].transform(X[col])
                    if isinstance(encoded, pd.DataFrame):
                        for i, col_name in enumerate(encoded.columns):
                            X[f'{col}_{col_name}'] = encoded[col_name]
                    else:
                        X[f'{col}_encoded'] = encoded
                    X = X.drop(columns=[col])
        
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values using various strategies."""
    
    def __init__(self, numerical_strategy: str = 'mean', categorical_strategy: str = 'most_frequent'):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_imputer_ = None
        self.categorical_imputer_ = None
        
    def fit(self, X, y=None):
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create imputers
        if numerical_cols:
            if self.numerical_strategy == 'knn':
                self.numerical_imputer_ = KNNImputer(n_neighbors=5)
            else:
                self.numerical_imputer_ = SimpleImputer(strategy=self.numerical_strategy)
            self.numerical_imputer_.fit(X[numerical_cols])
        
        if categorical_cols:
            self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer_.fit(X[categorical_cols])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute numerical columns
        if numerical_cols and self.numerical_imputer_ is not None:
            X[numerical_cols] = self.numerical_imputer_.transform(X[numerical_cols])
        
        # Impute categorical columns
        if categorical_cols and self.categorical_imputer_ is not None:
            X[categorical_cols] = self.categorical_imputer_.transform(X[categorical_cols])
        
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features."""
    
    def __init__(self, method: str = 'standard', numerical_cols: Optional[List[str]] = None):
        self.method = method
        self.numerical_cols = numerical_cols
        self.scaler_ = None
        
    def fit(self, X, y=None):
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler_.fit(X[self.numerical_cols])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.numerical_cols] = self.scaler_.transform(X[self.numerical_cols])
        return X


class WOEFeatureEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoding for categorical features."""
    
    def __init__(self, categorical_cols: List[str], target_col: str = 'FraudResult'):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.woe_encoders_ = {}
        
    def fit(self, X, y=None):
        if not WOE_AVAILABLE:
            raise ImportError("WOE package not available. Install with: pip install woe")
        
        for col in self.categorical_cols:
            if col in X.columns:
                self.woe_encoders_[col] = WOE()
                self.woe_encoders_[col].fit(X[col], y)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.categorical_cols:
            if col in X.columns and col in self.woe_encoders_:
                woe_values = self.woe_encoders_[col].transform(X[col])
                X[f'{col}_woe'] = woe_values
                X = X.drop(columns=[col])
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select the best features using statistical tests."""
    
    def __init__(self, k: int = 50, score_func=f_classif):
        self.k = k
        self.score_func = score_func
        self.selector_ = None
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            self.selector_ = SelectKBest(score_func=self.score_func, k=min(self.k, len(numerical_cols)))
            self.selector_.fit(X[numerical_cols], y)
        return self
    
    def transform(self, X):
        X = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols and self.selector_ is not None:
            selected_features = self.selector_.get_support()
            selected_cols = [col for col, selected in zip(numerical_cols, selected_features) if selected]
            X = X[selected_cols + [col for col in X.columns if col not in numerical_cols]]
        
        return X


def create_feature_engineering_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str],
    datetime_col: str = 'TransactionStartTime',
    customer_col: str = 'CustomerId',
    amount_col: str = 'Amount',
    target_col: str = 'FraudResult',
    encoding_strategy: str = 'onehot',
    scaling_method: str = 'standard',
    feature_selection_k: int = 50
) -> Pipeline:
    """
    Create a comprehensive feature engineering pipeline.
    
    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        datetime_col: Name of datetime column
        customer_col: Name of customer ID column
        amount_col: Name of amount column
        target_col: Name of target column
        encoding_strategy: Categorical encoding strategy ('onehot', 'label', 'target', 'woe')
        scaling_method: Feature scaling method ('standard', 'minmax')
        feature_selection_k: Number of features to select
    
    Returns:
        sklearn Pipeline with all feature engineering steps
    """
    
    # Define the pipeline steps
    steps = []
    
    # Step 1: Handle missing values
    steps.append(('missing_values', MissingValueHandler()))
    
    # Step 2: Extract temporal features
    if datetime_col:
        steps.append(('temporal_features', TemporalFeatureExtractor(datetime_col=datetime_col)))
    
    # Step 3: Create aggregate features
    if customer_col and amount_col:
        steps.append(('aggregate_features', AggregateFeatureExtractor(
            customer_col=customer_col, amount_col=amount_col
        )))
    
    # Step 4: Encode categorical features
    if categorical_cols:
        steps.append(('categorical_encoding', CategoricalFeatureEncoder(
            categorical_cols=categorical_cols, encoding_strategy=encoding_strategy
        )))
    
    # Step 5: Scale numerical features
    steps.append(('scaling', FeatureScaler(method=scaling_method)))
    
    # Step 6: Feature selection
    if feature_selection_k > 0:
        steps.append(('feature_selection', FeatureSelector(k=feature_selection_k)))
    
    return Pipeline(steps)


def create_advanced_feature_engineering_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str],
    datetime_col: str = 'TransactionStartTime',
    customer_col: str = 'CustomerId',
    amount_col: str = 'Amount',
    target_col: str = 'FraudResult'
) -> Pipeline:
    """
    Create an advanced feature engineering pipeline with WOE encoding and monotonic binning.
    
    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        datetime_col: Name of datetime column
        customer_col: Name of customer ID column
        amount_col: Name of amount column
        target_col: Name of target column
    
    Returns:
        sklearn Pipeline with advanced feature engineering steps
    """
    
    # Define the pipeline steps
    steps = []
    
    # Step 1: Handle missing values
    steps.append(('missing_values', MissingValueHandler()))
    
    # Step 2: Extract temporal features
    if datetime_col:
        steps.append(('temporal_features', TemporalFeatureExtractor(datetime_col=datetime_col)))
    
    # Step 3: Create aggregate features
    if customer_col and amount_col:
        steps.append(('aggregate_features', AggregateFeatureExtractor(
            customer_col=customer_col, amount_col=amount_col
        )))
    
    # Step 4: WOE encoding for categorical features (if available)
    if categorical_cols and WOE_AVAILABLE:
        steps.append(('woe_encoding', WOEFeatureEncoder(
            categorical_cols=categorical_cols, target_col=target_col
        )))
    elif categorical_cols:
        # Fallback to regular encoding
        steps.append(('categorical_encoding', CategoricalFeatureEncoder(
            categorical_cols=categorical_cols, encoding_strategy='label'
        )))
    
    # Step 5: Monotonic binning for numerical features (if available)
    if numerical_cols and XVERSE_AVAILABLE:
        steps.append(('monotonic_binning', MonotonicBinning()))
    
    # Step 6: Scale numerical features
    steps.append(('scaling', FeatureScaler(method='standard')))
    
    return Pipeline(steps)


def get_feature_importance_info(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Get feature importance information from the pipeline.
    
    Args:
        pipeline: Fitted feature engineering pipeline
        X: Input features
        y: Target variable
    
    Returns:
        Dictionary with feature importance information
    """
    
    # Transform the data
    X_transformed = pipeline.transform(X)
    
    # Get feature names
    feature_names = X_transformed.columns.tolist()
    
    # Calculate correlation with target
    correlations = {}
    for col in feature_names:
        if col in X_transformed.columns:
            corr = X_transformed[col].corr(y)
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return {
        'feature_names': feature_names,
        'correlations': dict(sorted_correlations),
        'top_features': [f[0] for f in sorted_correlations[:20]],
        'transformed_shape': X_transformed.shape
    }


def save_feature_engineering_pipeline(pipeline: Pipeline, filepath: str):
    """Save the feature engineering pipeline to disk."""
    import joblib
    joblib.dump(pipeline, filepath)


def load_feature_engineering_pipeline(filepath: str) -> Pipeline:
    """Load the feature engineering pipeline from disk."""
    import joblib
    return joblib.load(filepath) 