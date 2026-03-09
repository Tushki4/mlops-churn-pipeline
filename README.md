# mlops-churn-pipeline

## Dataset · IBM Telco Customer Churn

| Metric | Value |
|--------|-------|
| Records | 7,043 customers |
| Features | 20 (demographic + services + billing) |
| Target | Churn (Yes/No) — binary classification |
| Churn rate | 26.5% — moderately imbalanced |

### Key EDA Findings

**Strongest predictors of churn:**
- Contract type: month-to-month customers churn at 42.7%, vs ~11% (1yr) and ~3% (2yr)
- Tenure: churned customers average 17 months, retained average 37 months
- MonthlyCharges: churned avg $74, retained avg $61
- Fiber optic internet: highest churn among internet service types

**Data quality issues fixed:**
- TotalCharges stored as object dtype — 11 nulls for new customers (tenure=0)
- Fixed: pd.to_numeric(errors='coerce') then .fillna(0)

**Metric choice rationale:**
A model that always predicts no churn achieves 73.5% accuracy — useless.
We use ROC-AUC and Recall instead. Accuracy is not reported.