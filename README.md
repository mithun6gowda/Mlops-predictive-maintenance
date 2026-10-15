# Predictive Maintenance with MLOps

This project started with a simple question: **can machine sensor readings help us identify *why* a machine might fail, rather than just whether it will fail?**

I built a multiclass predictive maintenance workflow using the AI4I-style machine operating data provided for this assignment. Along the way, I wanted to practice the parts of machine learning that go beyond fitting a model: checking incoming data, dealing with rare failure events, tracking experiments, tuning a model, watching for data drift, and explaining predictions in engineering terms.

The final model is a tuned XGBoost classifier. It achieved a **0.7622 macro F1-score** on the held-out validation set. That is an improvement over the baseline XGBoost score of **0.7481**, although performance on Tool Wear Failure (TWF) is still a clear weakness.

## What the model predicts

The target has five classes:

| Label | Meaning |
|---|---|
| 0 | No Failure |
| 1 | Tool Wear Failure (TWF) |
| 2 | Heat Dissipation Failure (HDF) |
| 3 | Power Failure (PWF) |
| 4 | Overstrain Failure (OSF) |

The training dataset contains **6,993 rows**, with **1,499 rows** each in the current and stress datasets. The biggest challenge is class imbalance: approximately **96.7%** of training examples are labeled No Failure, while there are only **30 original TWF examples**.

## Workflow

### 1. Data checks and feature engineering

I used **Pandera** to validate the input datasets and explored class distributions before training. In addition to the original machine readings, I created two features:

- `Power_W`: mechanical power calculated from rotational speed and torque, using `2 × π × RPM × Torque / 60`.
- `Temp_diff`: process temperature minus air temperature.

The model uses eight features: `Type_enc`, `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`, `Power_W`, and `Temp_diff`.

Machine type was encoded consistently using a label encoder fitted on the training data. The encoder is saved so the same mapping can be reused.

### 2. Model training and experiment tracking

I made a stratified **80/20 train-validation split** and applied **SMOTE only to the training portion** (`k_neighbors=3`). Keeping the validation set untouched was important: oversampling it would have made the evaluation less representative.

I trained four models and tracked the experiments in **MLflow**:

| Model | Validation macro F1 |
|---|---:|
| Logistic Regression | 0.5312 |
| Random Forest | 0.7355 |
| LightGBM | 0.7296 |
| XGBoost | **0.7481** |

I selected the best baseline using **macro F1**, not accuracy. With so many No Failure records, a high accuracy score can hide poor performance on the rare failure classes.

Next, I ran **20 Optuna trials** to tune XGBoost. The tuned model reached:

| Metric | Tuned XGBoost |
|---|---:|
| Macro F1 | **0.7622** |
| Weighted F1 | 0.9857 |
| Accuracy | 0.9836 |

The tuned model was logged and registered in MLflow as `PredictiveMaintenance_XGBoost` (version 1). I also exported the trained model as `best_model.pkl` and the fitted type encoder as `label_encoder.pkl`.

**What did not work as well:** TWF remained difficult to identify. Its tuned-model F1-score was approximately **0.11**. With only 30 real TWF examples in the original training set, synthetic oversampling alone is not enough to solve that problem. This model needs more real minority-class data and further evaluation before anyone should rely on it for maintenance decisions.

### 3. Data drift monitoring

I used **Evidently** to compare the original training data against two later batches, using the same eight model input features.

| Comparison | Drifted features | Drift share |
|---|---:|---:|
| Training vs. current | 0 / 8 | 0% |
| Training vs. stress | 4 / 8 | 50% |

The stress dataset showed drift in:

| Feature | Drift score |
|---|---:|
| Rotational speed | 0.2354 |
| Torque | 0.4739 |
| Tool wear | 0.6455 |
| Power_W | 0.5199 |

All four exceeded the configured drift threshold of **0.1**. This tells me the stress batch differs meaningfully from the training distribution. It does **not**, on its own, prove that the classifier's accuracy has dropped; that would need labeled evaluation data.

The interactive HTML reports are saved as `drift_current.html` and `drift_stress.html`.

### 4. Explaining predictions with SHAP

I used `shap.TreeExplainer` on the saved XGBoost model and plotted mean absolute SHAP values separately for the four failure types. I kept the plots class-specific because one overall feature ranking would miss important differences between failure mechanisms.

| Failure class | Top SHAP feature | Engineering interpretation |
|---|---|---|
| TWF | Tool wear | Wear progression is strongly associated with this prediction |
| HDF | Temp_diff | Temperature separation is important for identifying heat dissipation problems |
| PWF | Power_W | Combined torque and speed carry more information than either alone |
| OSF | Tool wear | Wear, together with loading conditions, helps explain overstrain predictions |

The four-panel chart is saved as `shap_per_class.png`. These explanations describe model behavior, not proven physical causation.

## Project outputs

The main artifacts produced by the notebook are:

```text
eda_distributions.png   # Exploratory data analysis plots
best_model.pkl          # Tuned XGBoost model
label_encoder.pkl       # Fitted machine-type encoder
drift_current.html      # Current-data drift report
drift_stress.html       # Stress-data drift report
shap_per_class.png      # Four class-specific SHAP bar charts
mlflow.db               # Local MLflow tracking database (if retained)
```

The notebook also contains data validation, model comparisons, tuning results, drift analysis, and final engineering recommendations.

## Running the project

The workflow was developed in a local Python/Jupyter environment. You will need the project notebook and the three input CSV files (`train.csv`, `current.csv`, and `stress.csv`).

1. Create and activate a Python virtual environment.
2. Install the libraries used in the notebook: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `pandera`, `matplotlib`, `xgboost`, `lightgbm`, `mlflow`, `optuna`, `evidently`, `shap`, and `joblib`.
3. Place the CSV files where the notebook expects them, or update the data-loading paths.
4. Open the notebook in Jupyter and run the cells from top to bottom. The notebook generates the model, tracking records, plots, and drift reports.

For local experiment tracking, the notebook uses an SQLite-backed MLflow store (`sqlite:///mlflow.db`). If you want to inspect the runs in the MLflow UI, you can start it from the project directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local address printed by MLflow.

Package APIs can change between releases, especially Evidently and SHAP, so use a compatible environment when reproducing the notebook. The completed analysis used Evidently **0.7.23** and SHAP **0.51.0**.

## What I would improve next

The first priority would be **better detection of rare failures**, particularly TWF. I would collect more real failure examples, compare class-specific decision thresholds, and review confusion matrices and per-class precision/recall rather than relying on a single summary metric.

I would also test the model on a separate, time-ordered dataset, connect drift alerts to a monitoring workflow, and define maintenance thresholds with engineering input. The current project demonstrates an end-to-end MLOps workflow, but it is **not a production-certified maintenance system**.

## Main takeaway

This project showed me that getting a good overall score is only one part of predictive maintenance. The difficult work is recognizing rare failures, understanding when incoming data has changed, and making the model's decisions useful to someone maintaining the equipment. XGBoost gave the strongest overall validation result here, but the TWF limitation and stress-data drift are just as important as the final score.
