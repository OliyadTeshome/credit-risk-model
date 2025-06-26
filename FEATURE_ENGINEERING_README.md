# Feature Engineering Pipeline for Credit Risk Model

## Overview

This repository contains a comprehensive, automated, and reproducible feature engineering pipeline for credit risk modeling. The pipeline transforms raw transaction data into model-ready features using advanced techniques including Weight of Evidence (WOE) encoding, temporal feature extraction, and customer-level aggregations.

## 🚀 Features

### ✅ Implemented Transformations

1. **Aggregate Features**
   - Total transaction amount per customer
   - Average transaction amount per customer
   - Transaction count per customer
   - Standard deviation of transaction amounts per customer
   - Customer lifetime value and transaction frequency

2. **Temporal Features**
   - Transaction hour, day, month, year
   - Day of week, day of year, week number
   - Cyclical encoding (sin/cos) for periodic features
   - Business hours, weekend, and night indicators

3. **Categorical Encoding**
   - Label Encoding
   - One-Hot Encoding
   - Target Encoding
   - **Weight of Evidence (WOE) Encoding**
   - Information Value (IV) calculation

4. **Missing Value Handling**
   - Mean/median/mode imputation
   - KNN imputation
   - Forward/backward fill strategies

5. **Feature Scaling & Normalization**
   - StandardScaler (z-score normalization)
   - MinMaxScaler (0-1 normalization)
   - Robust scaling for outlier handling

6. **Advanced Techniques**
   - Monotonic binning (using xverse)
   - Feature selection using statistical tests
   - Correlation analysis and multicollinearity detection

## 📁 Project Structure

```
credit-risk-model/
├── src/
│   ├── feature_engineering.py      # Core feature engineering classes
│   ├── data_processing.py          # Data processing pipeline
│   ├── run_feature_engineering.py  # Example usage script
│   └── ...
├── tests/
│   ├── test_feature_engineering.py # Comprehensive tests
│   └── ...
├── data/
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data output
├── requirements.txt                # Dependencies
└── FEATURE_ENGINEERING_README.md   # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd credit-risk-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import xverse, woe, category_encoders; print('✅ All packages installed successfully!')"
   ```

## 📊 Usage

### Quick Start

```python
from src.data_processing import process_data_pipeline

# Run the complete pipeline
X_processed, y, processing_info = process_data_pipeline(
    data_path="data/raw/data.csv",
    output_path="data/processed/processed_data.csv",
    pipeline_type="advanced",
    encoding_strategy="woe",
    scaling_method="standard",
    feature_selection_k=50
)

print(f"✅ Processed {X_processed.shape[1]} features from raw data")
```

### Advanced Usage

```python
from src.feature_engineering import create_advanced_feature_engineering_pipeline
from src.data_processing import load_data, identify_feature_types

# Load data
df = load_data("data/raw/data.csv")

# Identify feature types
feature_types = identify_feature_types(df)

# Create advanced pipeline
pipeline = create_advanced_feature_engineering_pipeline(
    categorical_cols=feature_types['categorical'],
    numerical_cols=feature_types['numerical'],
    datetime_col='TransactionStartTime',
    customer_col='CustomerId',
    amount_col='Amount',
    target_col='FraudResult'
)

# Fit and transform
X = df.drop(columns=['FraudResult'])
y = df['FraudResult']

pipeline.fit(X, y)
X_processed = pipeline.transform(X)
```

### Running the Example Script

```bash
cd src
python run_feature_engineering.py
```

This will:
1. Load and validate the raw data
2. Run both standard and advanced pipelines
3. Generate visualizations and reports
4. Save processed data and pipeline artifacts

## 🔧 Pipeline Components

### 1. TemporalFeatureExtractor

Extracts comprehensive temporal features from datetime columns:

```python
from src.feature_engineering import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor(datetime_col='TransactionStartTime')
df_with_temporal = extractor.transform(df)

# Creates features like:
# - transaction_hour, transaction_day, transaction_month
# - hour_sin, hour_cos (cyclical encoding)
# - is_business_hours, is_weekend, is_night
```

### 2. AggregateFeatureExtractor

Creates customer-level aggregate features:

```python
from src.feature_engineering import AggregateFeatureExtractor

extractor = AggregateFeatureExtractor(customer_col='CustomerId', amount_col='Amount')
extractor.fit(df)
df_with_aggregates = extractor.transform(df)

# Creates features like:
# - customer_Amount_count, customer_Amount_total, customer_Amount_mean
# - Amount_to_customer_mean, Amount_to_customer_std
```

### 3. WOEFeatureEncoder

Applies Weight of Evidence encoding to categorical variables:

```python
from src.feature_engineering import WOEFeatureEncoder

encoder = WOEFeatureEncoder(categorical_cols=['ProductCategory', 'ProviderId'])
encoder.fit(X, y)
X_encoded = encoder.transform(X)

# Creates WOE-encoded features with information value
```

### 4. MissingValueHandler

Handles missing values using various strategies:

```python
from src.feature_engineering import MissingValueHandler

handler = MissingValueHandler(
    numerical_strategy='mean',
    categorical_strategy='most_frequent'
)
handler.fit(df)
df_imputed = handler.transform(df)
```

## 📈 Feature Engineering Techniques

### Weight of Evidence (WOE) and Information Value (IV)

WOE encoding transforms categorical variables into numerical values that represent the relationship between the feature and the target variable:

```python
# WOE Formula:
# WOE = ln(% of Good / % of Bad)
# IV = Σ (% of Good - % of Bad) * WOE

# Interpretation:
# IV < 0.02: Useless
# 0.02-0.1: Weak
# 0.1-0.3: Medium
# 0.3-0.5: Strong
# > 0.5: Suspicious (overfitting risk)
```

### Cyclical Encoding

For temporal features, we use cyclical encoding to preserve the periodic nature:

```python
# For hour (0-23):
hour_sin = sin(2 * π * hour / 24)
hour_cos = cos(2 * π * hour / 24)

# This ensures that hour 23 and hour 0 are close in feature space
```

### Customer-Level Aggregations

We create features that capture customer behavior patterns:

```python
# Transaction patterns
customer_transaction_count = len(customer_transactions)
customer_total_amount = sum(customer_transactions['Amount'])
customer_avg_amount = customer_total_amount / customer_transaction_count

# Temporal patterns
customer_lifetime_days = max_date - min_date
customer_transaction_frequency = transaction_count / lifetime_days

# Risk indicators
amount_to_customer_mean_ratio = transaction_amount / customer_avg_amount
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_feature_engineering.py -v

# Run specific test class
pytest tests/test_feature_engineering.py::TestTemporalFeatureExtractor -v

# Run with coverage
pytest tests/test_feature_engineering.py --cov=src --cov-report=html
```

## 📊 Output and Reports

The pipeline generates several outputs:

1. **Processed Data**: CSV files with engineered features
2. **Pipeline Artifacts**: Saved sklearn pipelines for reproducibility
3. **Visualizations**: Feature importance plots and comparisons
4. **Reports**: Comprehensive analysis reports

### Example Report Structure

```
FEATURE ENGINEERING REPORT
==========================

1. RAW DATA SUMMARY
   - Dataset shape: (95,662, 16)
   - Memory usage: 11.7 MB
   - Missing values: 0
   - Feature types: categorical(11), numerical(4), datetime(1)

2. STANDARD PIPELINE RESULTS
   - Original features: 16
   - Processed features: 45
   - Top features: customer_Amount_mean, transaction_hour, ...

3. ADVANCED PIPELINE RESULTS
   - Original features: 16
   - Processed features: 52
   - Top features: ProductCategory_woe, customer_Amount_std, ...

4. RECOMMENDATIONS
   - Use advanced pipeline for better feature representation
   - Focus on temporal and customer aggregate features
   - Monitor WOE stability across time periods
```

## 🔍 Feature Analysis

### Feature Importance Analysis

The pipeline automatically calculates feature importance using correlation analysis:

```python
from src.feature_engineering import get_feature_importance_info

feature_info = get_feature_importance_info(pipeline, X, y)
print("Top features:", feature_info['top_features'][:10])
```

### Correlation Analysis

```python
# Check for multicollinearity
correlation_matrix = X_processed.corr()
high_corr_pairs = np.where(np.abs(correlation_matrix) > 0.8)
```

## 🚨 Best Practices

### 1. Data Validation
- Always validate raw data before processing
- Check for data quality issues (missing values, duplicates)
- Verify feature types and distributions

### 2. Pipeline Reproducibility
- Save fitted pipelines for consistent transformations
- Use version control for pipeline configurations
- Document all preprocessing steps

### 3. Feature Stability
- Monitor feature distributions over time
- Validate WOE stability for categorical features
- Check for data drift in production

### 4. Performance Optimization
- Use appropriate data types (categorical, datetime)
- Implement efficient aggregation strategies
- Consider parallel processing for large datasets

## 🔧 Configuration

### Pipeline Configuration Options

```python
# Standard pipeline
pipeline_config = {
    'encoding_strategy': 'label',      # 'label', 'onehot', 'target', 'woe'
    'scaling_method': 'standard',      # 'standard', 'minmax', 'robust'
    'feature_selection_k': 50,         # Number of features to select
    'numerical_strategy': 'mean',      # Missing value strategy
    'categorical_strategy': 'most_frequent'
}

# Advanced pipeline
advanced_config = {
    'use_woe': True,                   # Enable WOE encoding
    'use_monotonic_binning': True,     # Enable monotonic binning
    'feature_selection': True,         # Enable feature selection
    'correlation_threshold': 0.8       # Multicollinearity threshold
}
```

## 📚 Dependencies

### Core Dependencies
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `scikit-learn` >= 1.0.0

### Advanced Feature Engineering
- `xverse` >= 0.2.0 (for monotonic binning)
- `woe` >= 0.1.0 (for WOE encoding)
- `category_encoders` >= 2.3.0 (for target encoding)

### Visualization and Testing
- `matplotlib` >= 3.4.0
- `seaborn` >= 0.11.0
- `pytest` >= 6.0.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This feature engineering pipeline is designed for credit risk modeling but can be adapted for other domains by modifying the feature extraction logic and target variable handling. 