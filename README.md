# Credit Risk Model - Machine Learning Pipeline

A comprehensive machine learning pipeline for credit risk assessment, featuring data processing, model training with MLflow tracking, containerized API deployment, and CI/CD automation.

## 🚀 Features

- **Data Processing**: Automated feature engineering and data preprocessing
- **Model Training**: Multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for model versioning and experiment management
- **API Service**: FastAPI-based REST API for real-time predictions
- **Containerization**: Docker support for easy deployment
- **CI/CD**: Automated testing and code quality checks
- **Unit Testing**: Comprehensive test coverage

## 📋 Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd credit-risk-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import mlflow, fastapi, sklearn; print('All dependencies installed successfully!')"
   ```

## 📊 Data

The project uses processed customer data with the following features:
- `recency_days`: Days since last transaction
- `frequency`: Number of transactions
- `monetary_total`: Total transaction amount
- `monetary_avg`: Average transaction amount
- `cluster`: Customer segment cluster
- `is_high_risk`: Target variable (0 = low risk, 1 = high risk)

## 🏃‍♂️ Quick Start

### 1. Train the Model

```bash
python src/train.py
```

This will:
- Load processed data from `data/processed/customer_risk_target.csv`
- Train multiple models with hyperparameter tuning
- Track experiments with MLflow
- Save the best model to `models/credit_risk_model.joblib`
- Register the model in MLflow Model Registry

### 2. Run the API

```bash
docker-compose up --build
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Test the API

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"recency_days": 10, "frequency": 5, "monetary_total": 20000, "monetary_avg": 4000, "cluster": 1}'

# Using PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"recency_days": 10, "frequency": 5, "monetary_total": 20000, "monetary_avg": 4000, "cluster": 1}'
```

## 📚 API Documentation

### Endpoints

#### POST `/predict`
Predict credit risk for a customer.

**Request Body:**
```json
{
  "recency_days": 10,
  "frequency": 5,
  "monetary_total": 20000,
  "monetary_avg": 4000,
  "cluster": 1
}
```

**Response:**
```json
{
  "risk_score": 0.1234
}
```

**Field Descriptions:**
- `recency_days` (float): Days since last transaction
- `frequency` (float): Number of transactions
- `monetary_total` (float): Total transaction amount
- `monetary_avg` (float): Average transaction amount
- `cluster` (int): Customer segment (0 or 1)
- `risk_score` (float): Probability of high risk (0-1)

## 🧪 Testing

### Run All Tests
```bash
# Set PYTHONPATH for imports
$env:PYTHONPATH='.'  # PowerShell
export PYTHONPATH='.'  # Bash

# Run tests
python -m pytest tests/ -v
```

### Run Linting
```bash
flake8 src tests
```

### Run Tests and Linting (CI/CD)
```bash
# This is what GitHub Actions runs
flake8 src tests
$env:PYTHONPATH='.'; python -m pytest tests/ -v
```

## 📈 MLflow Integration

### View Experiments
```bash
mlflow ui
```
Open http://localhost:5000 to view:
- Experiment runs and metrics
- Model versions and registry
- Artifacts and parameters

### Model Registry
The best model is automatically registered as `CreditRiskBestModel` in the MLflow Model Registry.

## 🐳 Docker Deployment

### Local Development
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Deployment

1. **Build the image**
   ```bash
   docker build -t credit-risk-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 credit-risk-api
   ```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. **Triggers on**: Push to main branch and pull requests
2. **Runs on**: Ubuntu latest
3. **Steps**:
   - Set up Python 3.10
   - Install dependencies
   - Run flake8 linting
   - Run pytest tests
   - Fails if any step fails

## 📁 Project Structure

```
credit-risk-model/
├── data/
│   └── processed/
│       └── customer_risk_target.csv
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── pydantic_models.py   # Request/response models
│   │   └── data_processing.py   # Data processing utilities
│   │   └── feature_engineering.py # Feature engineering pipeline
│   │   └── train.py             # Model training script
│   │   └── predict.py           # Prediction utilities
│   ├── tests/
│   │   ├── test_data_processing.py  # Data processing tests
│   │   └── test_feature_engineering.py  # Feature engineering tests
│   ├── models/                      # Saved models
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Container configuration
│   ├── docker-compose.yml           # Local development setup
│   └── .github/workflows/ci.yml     # CI/CD pipeline
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Set PYTHONPATH
   $env:PYTHONPATH='.'  # PowerShell
   export PYTHONPATH='.'  # Bash
   ```

2. **Docker Build Fails**
   ```bash
   # Clear Docker cache
   docker system prune -a
   docker-compose up --build
   ```

3. **MLflow Model Not Found**
   - Ensure you've run `python src/train.py` first
   - Check that `models/credit_risk_model.joblib` exists

4. **API Connection Refused**
   ```bash
   # Check if container is running
   docker ps
   
   # Restart if needed
   docker-compose down
   docker-compose up --build
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub

---

**Happy coding! 🎉** 