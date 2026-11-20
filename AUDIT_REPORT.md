# Project Audit — Insurance Claim Fraud Detection

**Audit scope:** Uploaded project ZIP, code modules, model artifact, notebooks, reports, data, dashboard and tests.  
**Audit date:** 20 November 2026  
**Audit type:** Portfolio/reproducibility audit—not production certification.

## Executive verdict

**Status: portfolio-ready after documentation/dependency hygiene; not production-ready.** The project is an end-to-end working prototype with a modular pipeline, trained artifact, dashboard and automated tests. The most important limitations are the historical dataset, random split instead of temporal validation, low recall at the default threshold, no calibration/fairness/operational impact validation, and serialized-model dependency compatibility.

## Verification performed

- Inspected ZIP contents while excluding the bundled `.venv` and cache directories from the audit extraction.
- Confirmed raw data dimensions and target prevalence.
- Reviewed reusable source modules and dashboard calculations.
- Reviewed training metadata, notebook experiment metrics, project README, requirements and `.gitignore`.
- Ran `python -m pytest -q`: **8 passed**.
- Python dashboard syntax compilation: passed.
- Scoring tests emitted warnings because the serialized sklearn estimators were created under **scikit-learn 1.9.1**, while the audit environment loaded them under **1.8.0**.

## Findings

### P0/P1 — dependency compatibility for model artifact (high priority)

- `fraud_model_pipeline.joblib` was created with scikit-learn 1.9.1.
- The audit runtime used scikit-learn 1.8.0 and emitted `InconsistentVersionWarning` for several estimators.
- **Action taken:** pinned `scikit-learn==1.9.1` in the updated `requirements.txt`.
- Rebuild the environment from requirements and rerun tests before deployment. Joblib/pickle artifacts should not be loaded from untrusted sources.

### P1 — validation methodology

- Current train/test split is random and stratified.
- Claims may contain time-related fields, but the project does not perform temporal holdout validation.
- **Action:** temporal validation and rolling backtests are listed as required next improvements.

### P1 — model performance/interpretation

- At threshold 0.50, held-out recall is 4.86% in the reusable training report.
- Top-40 precision is 40.0%, recall@40 is 8.65%, and precision lift over random is about 6.67×.
- The model ranking improves over random selection, but absolute capture is limited.
- Scores are not shown to be calibrated probabilities.
- **Action:** README distinguishes score, threshold label and capacity ranking; future calibration and stronger validation remain open.

### P1 — responsible use/fairness

- Features include demographic variables, including Sex, Age and MaritalStatus.
- No subgroup fairness analysis or governance review is included.
- **Action:** README flags the risk and states human review must remain in the loop; fairness and feature-policy review is a future requirement.

### P2 — project hygiene

- Original `.gitignore` ignored all model artifacts, including the model required by the dashboard, and included an overly broad generated-data rule.
- `requirements.txt` contained duplicate `pytest` and unused `xgboost`/`shap` dependencies.
- ZIP included a local virtual environment and cache files.
- **Action taken:** updated `.gitignore` to preserve the demo pipeline artifact while ignoring experimental model files and generated scoring outputs; cleaned requirements to remove unused dependencies and pin sklearn; cleaned the updated archive of `.venv`, caches and compiled files.

### P2 — source attribution

- Data README says the file is from Kaggle, but the exact URL, author and license/terms are not recorded.
- **Action:** README explicitly identifies this gap and requests exact citation and terms before public redistribution/deployment.

### P2 — documentation and deployment

- Original README was a brief outline without setup, reproducibility, metrics, architecture, dashboard details, limitations or deployment instructions.
- **Action taken:** replaced with detailed README covering the business objective, assumptions, data audit, architecture, features, model results, metrics, dashboard, local setup, training/scoring, tests, Streamlit deployment, limitations and future work.

## Data audit

| Check | Result |
|---|---:|
| Rows | 15,420 |
| Columns | 33 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Fraud labels | 923 |
| Non-fraud labels | 14,497 |
| Fraud prevalence | 5.99% |

## Test results

**8 passed**. Test coverage includes schema/data loading, target/identifier separation, feature-engineering schema and leakage checks, inference schema validation, valid scoring outputs, capacity calculation, queue capacity/ranking, and invalid score rejection.

## Dashboard review

The uploaded dashboard contains the refreshed six-tab experience: Overview, Risk Analytics, Fraud Patterns, Investigation Queue, Claim Review and Export. Capacity utilization is calculated as selected queue / configured capacity; investigation coverage is calculated as selected queue / all claims. The uploaded code includes the earlier correction.

## Model result summary

From the current reusable training report:

| Metric | Value |
|---|---:|
| Accuracy | 94.07% |
| Precision at 0.50 | 56.25% |
| Recall at 0.50 | 4.86% |
| F1 at 0.50 | 8.96% |
| ROC-AUC | 0.8302 |
| Average Precision | 0.2369 |
| Top-K | K=40 on test |
| Fraud in Top-40 | 16 |
| Precision@40 | 40.00% |
| Recall@40 | 8.65% |
| Random baseline precision | ~6.00% |
| Lift | ~6.67× |

## Actions completed in this audit pass

- [x] Extracted and inspected project ZIP.
- [x] Verified data statistics.
- [x] Ran automated tests (8/8 pass).
- [x] Reviewed dashboard code and KPI calculation.
- [x] Pin scikit-learn to artifact version.
- [x] Removed duplicate/unused dependencies from requirements.
- [x] Updated `.gitignore` so the demo model artifact is not accidentally excluded.
- [x] Wrote detailed project README.
- [x] Documented risks, limitations and public-release checklist.
- [x] Created a cleaned project ZIP without virtual environment/cache directories.

## Recommended remaining sequence

1. Recreate environment from updated `requirements.txt` and rerun tests locally.
2. Confirm Git tracking of the model artifact and raw dataset is intentional and allowed by the dataset license.
3. Add the exact dataset source URL/attribution/license.
4. Add dashboard screenshots to the README.
5. Perform temporal validation and calibration/fairness review as portfolio extensions.
6. Deploy a demo using historical or synthetic data only.
7. Prepare a concise recruiter case study and interview walkthrough.

## Release decision

**Suitable for a public portfolio demo after source attribution and a clean dependency/test run. Not suitable for real claim decisions or production use.**
