"""
Tests for Feature Engineering Pipeline

This module contains comprehensive tests for the feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import (
    TemporalFeatureExtractor,
    AggregateFeatureExtractor,
    CategoricalFeatureEncoder,
    MissingValueHandler,
    FeatureScaler,
    WOEFeatureEncoder,
    FeatureSelector,
    create_feature_engineering_pipeline,
    create_advanced_feature_engineering_pipeline
)

from data_processing import (
    load_data,
    validate_data,
    identify_feature_types,
    create_transaction_features,
    create_customer_features,
    prepare_model_data,
    process_data_pipeline
)


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'TransactionStartTime': [
                '2018-12-24T16:30:13Z',
                '2018-11-15T07:03:26Z',
                '2018-12-07T13:09:44Z'
            ],
            'Amount': [100, 200, 300],
            'CustomerId': ['C1', 'C2', 'C1']
        })
    
    def test_temporal_feature_extraction(self):
        """Test that temporal features are correctly extracted."""
        extractor = TemporalFeatureExtractor()
        result = extractor.transform(self.df)
        
        # Check that temporal features are created
        expected_features = [
            'transaction_hour', 'transaction_day', 'transaction_month',
            'transaction_year', 'transaction_dayofweek', 'transaction_dayofyear',
            'transaction_week', 'transaction_quarter'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check cyclical features
        cyclical_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos'
        ]
        
        for feature in cyclical_features:
            assert feature in result.columns
        
        # Check business indicators
        business_features = ['is_business_hours', 'is_weekend', 'is_night']
        for feature in business_features:
            assert feature in result.columns
        
        # Check that original datetime column is removed
        assert 'TransactionStartTime' not in result.columns


class TestAggregateFeatureExtractor:
    """Test aggregate feature extraction."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
            'Amount': [100, 200, 150, 250, 300],
            'Value': [110, 220, 160, 260, 310]
        })
    
    def test_aggregate_feature_extraction(self):
        """Test that aggregate features are correctly created."""
        extractor = AggregateFeatureExtractor()
        extractor.fit(self.df)
        result = extractor.transform(self.df)
        
        # Check that customer aggregate features are created
        expected_features = [
            'customer_Amount_count', 'customer_Amount_total', 'customer_Amount_mean',
            'customer_Amount_std', 'customer_Amount_min', 'customer_Amount_max',
            'customer_Amount_median'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check ratio features
        ratio_features = [
            'Amount_to_customer_mean', 'Amount_to_customer_std'
        ]
        
        for feature in ratio_features:
            assert feature in result.columns
        
        # Check that customer C1 has correct aggregate values
        c1_data = result[result['CustomerId'] == 'C1']
        assert c1_data['customer_Amount_count'].iloc[0] == 2
        assert c1_data['customer_Amount_total'].iloc[0] == 300
        assert c1_data['customer_Amount_mean'].iloc[0] == 150


class TestCategoricalFeatureEncoder:
    """Test categorical feature encoding."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'Category1': ['A', 'B', 'A', 'C'],
            'Category2': ['X', 'Y', 'X', 'Z'],
            'Amount': [100, 200, 300, 400]
        })
    
    def test_label_encoding(self):
        """Test label encoding of categorical features."""
        encoder = CategoricalFeatureEncoder(['Category1', 'Category2'], 'label')
        encoder.fit(self.df)
        result = encoder.transform(self.df)
        
        # Check that categorical columns are encoded
        assert 'Category1' in result.columns
        assert 'Category2' in result.columns
        
        # Check that values are numeric
        assert result['Category1'].dtype in ['int64', 'int32']
        assert result['Category2'].dtype in ['int64', 'int32']


class TestMissingValueHandler:
    """Test missing value handling."""
    
    def setup_method(self):
        """Set up test data with missing values."""
        self.df = pd.DataFrame({
            'Numerical1': [1, 2, np.nan, 4],
            'Numerical2': [10, np.nan, 30, 40],
            'Categorical1': ['A', 'B', np.nan, 'D'],
            'Categorical2': ['X', np.nan, 'Z', 'W']
        })
    
    def test_missing_value_handling(self):
        """Test that missing values are properly handled."""
        handler = MissingValueHandler()
        handler.fit(self.df)
        result = handler.transform(self.df)
        
        # Check that no missing values remain
        assert result.isnull().sum().sum() == 0
        
        # Check that numerical columns are filled with mean
        assert result['Numerical1'].isnull().sum() == 0
        assert result['Numerical2'].isnull().sum() == 0
        
        # Check that categorical columns are filled with most frequent
        assert result['Categorical1'].isnull().sum() == 0
        assert result['Categorical2'].isnull().sum() == 0


class TestFeatureScaler:
    """Test feature scaling."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Categorical': ['A', 'B', 'C', 'D', 'E']
        })
    
    def test_standard_scaling(self):
        """Test standard scaling of numerical features."""
        scaler = FeatureScaler(method='standard')
        scaler.fit(self.df)
        result = scaler.transform(self.df)
        
        # Check that numerical features are scaled
        assert abs(result['Feature1'].mean()) < 1e-10  # Mean should be close to 0
        assert abs(result['Feature1'].std() - 1) < 1e-10  # Std should be close to 1
        
        # Check that categorical features are unchanged
        assert (result['Categorical'] == self.df['Categorical']).all()
    
    def test_minmax_scaling(self):
        """Test min-max scaling of numerical features."""
        scaler = FeatureScaler(method='minmax')
        scaler.fit(self.df)
        result = scaler.transform(self.df)
        
        # Check that numerical features are scaled to [0, 1]
        assert result['Feature1'].min() == 0
        assert result['Feature1'].max() == 1
        
        # Check that categorical features are unchanged
        assert (result['Categorical'] == self.df['Categorical']).all()


class TestFeatureSelector:
    """Test feature selection."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Feature1': np.random.randn(100),
            'Feature2': np.random.randn(100),
            'Feature3': np.random.randn(100),
            'Feature4': np.random.randn(100),
            'Feature5': np.random.randn(100)
        })
        self.y = np.random.randint(0, 2, 100)
    
    def test_feature_selection(self):
        """Test that feature selection works correctly."""
        selector = FeatureSelector(k=3)
        selector.fit(self.df, self.y)
        result = selector.transform(self.df)
        
        # Check that only 3 numerical features are selected
        numerical_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numerical_cols) == 3


class TestFeatureEngineeringPipeline:
    """Test complete feature engineering pipeline."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'TransactionId': [f'T{i}' for i in range(100)],
            'CustomerId': [f'C{i%10}' for i in range(100)],
            'Amount': np.random.randn(100) * 1000 + 1000,
            'Value': np.random.randn(100) * 1000 + 1000,
            'TransactionStartTime': pd.date_range('2018-01-01', periods=100, freq='H'),
            'ProductCategory': np.random.choice(['A', 'B', 'C'], 100),
            'ProviderId': np.random.choice(['P1', 'P2', 'P3'], 100),
            'FraudResult': np.random.randint(0, 2, 100)
        })
    
    def test_standard_pipeline_creation(self):
        """Test that standard pipeline can be created."""
        categorical_cols = ['ProductCategory', 'ProviderId']
        numerical_cols = ['Amount', 'Value']
        
        pipeline = create_feature_engineering_pipeline(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            datetime_col='TransactionStartTime',
            customer_col='CustomerId',
            amount_col='Amount',
            target_col='FraudResult'
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) > 0
    
    def test_advanced_pipeline_creation(self):
        """Test that advanced pipeline can be created."""
        categorical_cols = ['ProductCategory', 'ProviderId']
        numerical_cols = ['Amount', 'Value']
        
        pipeline = create_advanced_feature_engineering_pipeline(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            datetime_col='TransactionStartTime',
            customer_col='CustomerId',
            amount_col='Amount',
            target_col='FraudResult'
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) > 0
    
    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and transform data."""
        categorical_cols = ['ProductCategory', 'ProviderId']
        numerical_cols = ['Amount', 'Value']
        
        pipeline = create_feature_engineering_pipeline(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            datetime_col='TransactionStartTime',
            customer_col='CustomerId',
            amount_col='Amount',
            target_col='FraudResult'
        )
        
        # Prepare data
        X = self.df.drop(columns=['FraudResult'])
        y = self.df['FraudResult']
        
        # Fit and transform
        pipeline.fit(X, y)
        result = pipeline.transform(X)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)


class TestDataProcessing:
    """Test data processing functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'TransactionId': [f'T{i}' for i in range(50)],
            'CustomerId': [f'C{i%5}' for i in range(50)],
            'Amount': np.random.randn(50) * 1000 + 1000,
            'Value': np.random.randn(50) * 1000 + 1000,
            'TransactionStartTime': pd.date_range('2018-01-01', periods=50, freq='H'),
            'ProductCategory': np.random.choice(['A', 'B', 'C'], 50),
            'ProviderId': np.random.choice(['P1', 'P2', 'P3'], 50),
            'FraudResult': np.random.randint(0, 2, 50)
        })
    
    def test_validate_data(self):
        """Test data validation."""
        validation_report = validate_data(self.df)
        
        assert 'shape' in validation_report
        assert 'missing_values' in validation_report
        assert 'data_types' in validation_report
        assert 'duplicates' in validation_report
        assert 'quality_issues' in validation_report
    
    def test_identify_feature_types(self):
        """Test feature type identification."""
        feature_types = identify_feature_types(self.df)
        
        assert 'categorical' in feature_types
        assert 'numerical' in feature_types
        assert 'datetime' in feature_types
        assert 'id_columns' in feature_types
        assert 'target' in feature_types
        
        # Check that target is identified
        assert 'FraudResult' in feature_types['target']
        
        # Check that ID columns are identified
        assert 'TransactionId' in feature_types['id_columns']
        assert 'CustomerId' in feature_types['id_columns']
    
    def test_create_transaction_features(self):
        """Test transaction feature creation."""
        result = create_transaction_features(self.df)
        
        # Check that temporal features are created
        temporal_features = [
            'transaction_hour', 'transaction_day', 'transaction_month',
            'transaction_year', 'transaction_dayofweek', 'transaction_dayofyear'
        ]
        
        for feature in temporal_features:
            assert feature in result.columns
        
        # Check that cyclical features are created
        cyclical_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos'
        ]
        
        for feature in cyclical_features:
            assert feature in result.columns
    
    def test_create_customer_features(self):
        """Test customer feature creation."""
        result = create_customer_features(self.df)
        
        # Check that customer features are created
        assert 'CustomerId' in result.columns
        
        # Check that aggregate features are created
        aggregate_features = [
            'Amount_count', 'Amount_sum', 'Amount_mean',
            'Value_count', 'Value_sum', 'Value_mean'
        ]
        
        for feature in aggregate_features:
            assert feature in result.columns
    
    def test_prepare_model_data(self):
        """Test model data preparation."""
        X_train, y_train, X_test, y_test = prepare_model_data(self.df)
        
        # Check that data is split correctly
        assert len(X_train) + len(X_test) == len(self.df)
        assert len(y_train) + len(y_test) == len(self.df)
        
        # Check that target is separated
        assert 'FraudResult' not in X_train.columns
        assert 'FraudResult' not in X_test.columns


if __name__ == "__main__":
    pytest.main([__file__]) 