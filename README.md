# Credit Risk Model

A production-ready machine learning project for credit scoring based on customer transaction data. Exposes a REST API using FastAPI.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # GitHub Actions for CI/CD
├── data/                      # Raw and processed data (add to .gitignore)
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb          # EDA and initial analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # RFM feature engineering per CustomerId
│   ├── train.py               # Model training script (Random Forest)
│   ├── predict.py             # Model inference using joblib
│   └── api/
│       ├── main.py            # FastAPI app with /predict-risk endpoint
│       └── pydantic_models.py # Request/response schemas
├── tests/
│   └── test_data_processing.py # Unit test for feature engineering
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability

The Basel II Capital Accord fundamentally transformed banking regulation by introducing risk-based capital requirements. This regulatory framework emphasizes **accurate risk measurement** and **transparent risk assessment processes**. The accord's three pillars—minimum capital requirements, supervisory review, and market discipline—collectively demand that financial institutions demonstrate:

1. **Risk Quantification**: Banks must precisely measure credit risk exposure and maintain adequate capital reserves
2. **Model Validation**: All risk models must undergo rigorous validation and be approved by regulators
3. **Documentation Requirements**: Complete documentation of model methodology, assumptions, and limitations is mandatory
4. **Explainability**: Models must be interpretable to both regulators and senior management

This regulatory environment creates a critical need for **interpretable and well-documented models** because:
- Regulators require clear understanding of how risk is calculated
- Model decisions must be explainable to customers and stakeholders
- Audit trails are essential for compliance and risk management
- Model performance must be continuously monitored and reported

### Proxy Variables: Necessity and Business Risks

In real-world credit scoring scenarios, direct "default" labels are often unavailable due to:
- **Data Privacy Regulations**: Customer default information may be restricted
- **Reporting Delays**: Default events may not be immediately recorded
- **Definitional Challenges**: Different institutions define default differently
- **Data Availability**: Historical default data may be limited or incomplete

**Creating proxy variables becomes necessary** to approximate default risk using available behavioral and transactional data. Common proxies include:
- Payment delays (30+, 60+, 90+ days past due)
- Account closures or charge-offs
- Credit limit reductions
- Behavioral changes in spending patterns

**Potential Business Risks of Proxy-Based Predictions:**

1. **Model Drift**: Proxies may not accurately represent true default risk over time
2. **Regulatory Scrutiny**: Regulators may question the validity of proxy definitions
3. **Performance Degradation**: Model accuracy may decline if proxy-target relationship changes
4. **Fair Lending Concerns**: Proxy variables might introduce bias or discrimination
5. **Capital Adequacy**: Incorrect risk estimates could lead to insufficient capital reserves

### Model Complexity Trade-offs in Regulated Financial Context

**Simple, Interpretable Models (e.g., Logistic Regression with WoE)**

**Advantages:**
- **Regulatory Compliance**: Easier to explain and validate with regulators
- **Transparency**: Clear feature importance and decision logic
- **Stability**: More robust to data drift and external changes
- **Documentation**: Simpler to document and maintain
- **Audit Trail**: Straightforward to audit and verify decisions

**Disadvantages:**
- **Performance Limitations**: May not capture complex non-linear relationships
- **Feature Engineering Dependency**: Requires extensive manual feature engineering
- **Lower Predictive Power**: May achieve lower AUC/KS statistics

**Complex, High-Performance Models (e.g., Gradient Boosting, Neural Networks)**

**Advantages:**
- **Superior Performance**: Often achieve higher predictive accuracy
- **Automatic Feature Learning**: Can discover complex patterns automatically
- **Better Risk Differentiation**: More granular risk segmentation
- **Competitive Advantage**: May provide edge in risk-based pricing

**Disadvantages:**
- **Regulatory Challenges**: Difficult to explain to regulators and stakeholders
- **Black Box Problem**: Decision logic is not transparent
- **Validation Complexity**: More difficult to validate and monitor
- **Overfitting Risk**: May not generalize well to new data
- **Compliance Burden**: Requires extensive documentation and justification

**Recommended Approach for Regulated Context:**

1. **Start Simple**: Begin with interpretable models (Logistic Regression, Scorecards)
2. **Incremental Complexity**: Gradually introduce more complex models with proper validation
3. **Hybrid Approach**: Use complex models for feature engineering, simple models for final decisions
4. **Comprehensive Documentation**: Maintain detailed documentation regardless of model complexity
5. **Regular Validation**: Implement robust model monitoring and validation frameworks

The choice between model complexity should be driven by a careful balance of regulatory requirements, business needs, and risk tolerance, with interpretability often taking precedence in highly regulated financial environments.

## How to Run

### 1. Build and start the API using Docker Compose

```bash
cd credit-risk-model
# Build and start the service
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### 2. Example: Predict Credit Risk

Send a POST request to `/predict-risk`:

```json
{
  "recency_days": 10,
  "frequency": 5,
  "monetary": 1200.50
}
```

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/predict-risk" \
     -H "Content-Type: application/json" \
     -d '{"recency_days": 10, "frequency": 5, "monetary": 1200.50}'
```

Response:
```json
{
  "risk_score": 0.23
}
```

## Testing

Run unit tests with:
```bash
pytest
```

## License
MIT 