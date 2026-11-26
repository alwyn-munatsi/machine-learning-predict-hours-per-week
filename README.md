# The Quest to Predict Labor: Machine Learning Regression on Weekly Work Hours

[![Python](https://img.shields.io/badge/Language-Python%203-3776AB?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-189AB4)](https://xgboost.ai/)
[![statsmodels](https://img.shields.io/badge/Stats-statsmodels-4051B5)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org/)

An end-to-end machine learning study that models **weekly hours worked** (`hours_per_week`) as a continuous measure of labour supply, using the **UCI Adult (Census Income)** dataset. The notebook moves from univariate and bivariate EDA (Spearman correlation, group means, Tukey HSD) through cleaning and encoding, **ANOVA-based feature selection**, and **GridSearchCV tuning of five regression models** (Negative Binomial, KNN, SVR, Decision Tree, XGBoost), compares them with **5-fold cross-validation**, evaluates the winner on a held-out test set, and tests whether a **VotingRegressor ensemble** beats the best single model.

**Project Members:** Alwyn Munatsi, Lucia Shumba, Chidochashe Makanga, Bekithemba Nkomo · **Supervisor:** James Topor
**Course:** DAV 6150 — Final Project

---

## 1. Analytical Workflow & End-to-End Flow

### End-to-End Execution Flow
$$\text{UCI Adult CSV} \xrightarrow{\text{EDA}} \text{Insights} \xrightarrow{\text{Cleaning + Encoding}} \text{Feature matrix} \xrightarrow{\text{SelectKBest}} \text{5 features} \xrightarrow{\text{GridSearchCV}} \text{5 tuned models} \xrightarrow{\text{5-fold CV}} \text{Best model} \xrightarrow{} \text{Test set + Ensemble}$$

1. **Data ingestion:** The Adult training file (32,561 rows × 15 columns) is read from the `FinalProject_Data.csv` hosted in the [DAV-6150](https://github.com/alwyn-munatsi/DAV-6150) repository.
2. **EDA (Section 2):** Distributions of every variable, Spearman correlations for continuous features, mean hours by each category, Tukey HSD for native country, and a relationship × marital-status cross-tab.
3. **Data preparation (Section 3):** Drop redundant columns, fix `?` placeholders, binarise sparse and two-level variables, one-hot encode the rest.
4. **Prepared-data EDA (Section 4):** Re-check every engineered feature against `hours_per_week`.
5. **Model training (Section 6):** 80/20 train/test split, feature selection, then five tuned models.
6. **Model selection (Section 7):** 5-fold cross-validation on RMSE, MAE and R²; the best model is scored on the test set.
7. **Ensemble (Section 8):** A `VotingRegressor` of KNN, SVR and Decision Tree is compared with the best individual model.

---

## 2. Dataset & Variable Directory

| Attribute | Details |
| :--- | :--- |
| **Source** | [UCI Machine Learning Repository – Adult](https://archive.ics.uci.edu/dataset/2/adult) (1994 U.S. Census; 48,842 records across the original train/test files) |
| **Rows used** | 32,561 (the training file) |
| **Target** | `hours_per_week` (continuous) |
| **Numeric features** | `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss` |
| **Categorical features** | `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`, `income` |

### Research Questions
1. Which demographic, educational and occupational attributes most strongly influence weekly labour supply?
2. Can regression models accurately predict weekly hours worked from demographic and socioeconomic features?
3. Does an ensemble regressor outperform the individual models?

---

## 3. Modelling Decisions

| Decision | Choice | Justification |
| :--- | :--- | :--- |
| **Redundant columns** | Drop `fnlwgt`, `education`, `marital_status` | `fnlwgt` is a sampling weight; `education` duplicates `education_num`; `marital_status` overlaps heavily with `relationship` (confirmed by cross-tab) |
| **Sparse money columns** | `capital_gain` / `capital_loss` → `has_capital_gain` / `has_capital_loss` | Almost all values are zero; presence is more informative than amount |
| **Missing values** | `?` in `workclass` and `occupation` → `Unknown` | Keeps rows and lets "unknown" carry its own signal |
| **High-cardinality country** | Non-US countries → `Other`, then binary | Most records are US; rare countries add noise |
| **Two-level variables** | Binary-encode `sex`, `income`, `native_country` | Only two outcomes each |
| **Remaining categoricals** | One-hot encode `workclass`, `occupation`, `relationship`, `race` | Nominal with no natural order |
| **Feature selection** | `SelectKBest(f_classif, k=5)` | Keeps the five strongest ANOVA F-value features |
| **Count-aware baseline** | Negative Binomial (statsmodels) | Hours are non-negative integers with over-dispersion |
| **Scaling** | `StandardScaler` for KNN and SVR | Distance- and kernel-based models are scale-sensitive |
| **Negative predictions** | Clipped with `np.maximum(0, ...)` | Hours cannot be negative |

---

## 4. Phase 1: Exploratory Data Analysis

* **Univariate:** summary statistics, histograms and box plots for numeric columns; value counts and bar charts for categoricals.
* **Bivariate:** Spearman rank correlation heatmap; mean `hours_per_week` by education, workclass, marital status, occupation, relationship, race, sex, native country and income.
* **Post-hoc testing:** Tukey's HSD on native country vs. hours per week.

---

## 5. Phase 2: Data Cleaning & Feature Engineering

```python
# Drop redundant columns
df = df.drop(columns=['fnlwgt', 'education', 'marital_status'])

# Binary flags for sparse capital columns
df['has_capital_gain'] = (df['capital_gain'] > 0).astype(bool)
df['has_capital_loss'] = (df['capital_loss'] > 0).astype(bool)

# Replace '?' placeholders
df['workclass']  = df['workclass'].replace(' ?', 'Unknown')
df['occupation'] = df['occupation'].replace(' ?', 'Unknown')

# Group non-US countries
df['native_country'] = df['native_country'].apply(lambda x: 'Other' if x != ' United-States' else x)

# One-hot encode nominal variables
df = pd.get_dummies(df, columns=['workclass', 'occupation', 'relationship', 'race'])
```

---

## 6. Phase 3: Feature Selection & Model Training

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_classif, k=5)
selector.fit_transform(X, y)
# Selected: ['sex', 'income', 'occupation_Unknown', 'relationship_ Husband', 'relationship_ Own-child']
```

| Model | Library | Tuning Grid | Best Parameters |
| :--- | :--- | :--- | :--- |
| **Negative Binomial** | `statsmodels` | Dispersion estimated by MLE | — |
| **KNN Regressor** | scikit-learn | `n_neighbors` 3–11, `weights`, `p` ∈ {1, 2} | `n_neighbors=11`, `p=1`, `weights='uniform'` |
| **SVR** | scikit-learn | `kernel='rbf'`, `C` ∈ {0.1, 1}, `epsilon` ∈ {0.1, 0.2} | `C=1`, `epsilon=0.2` |
| **Decision Tree** | scikit-learn | `max_depth`, `min_samples_split`, `min_samples_leaf` | `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1` |
| **XGBoost** | `xgboost` | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {3, 5}, `learning_rate` ∈ {0.01, 0.1} | `n_estimators=100`, `max_depth=3`, `learning_rate=0.1` |

---

## 7. Phase 4: Cross-Validation & Model Selection

5-fold `KFold(shuffle=True, random_state=42)` on the training set (mean across folds):

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| KNN | 11.9291 | 8.3784 | 0.0637 |
| SVR | 11.7476 | **7.3425** | 0.0922 |
| Decision Tree | 11.3895 | 7.9077 | 0.1467 |
| **XGBoost** | **11.3870** | 7.9048 | **0.1471** |
| Negative Binomial | 11.4226 | 7.9473 | 0.1417 |

**XGBoost** was selected (lowest RMSE, highest R²).

### Held-out test set (XGBoost)
| Metric | Test | CV |
| :--- | :--- | :--- |
| RMSE | 11.4715 | 11.3870 |
| MAE | 7.9985 | 7.9048 |
| R² | 0.1456 | 0.1471 |

Test and CV scores are close, so the model generalises consistently.

---

## 8. Phase 5: Ensemble Model

```python
estimators = [
    ('knn', best_knn_model),
    ('svr', svr_ensemble_pipeline),   # StandardScaler + SVR
    ('dt',  best_dt_model),
]
ensemble_model = VotingRegressor(estimators=estimators, n_jobs=-1)
ensemble_model.fit(X_train[selected_features], y_train)
```

| Model (test set) | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **11.4715** | 7.9985 | **0.1456** |
| VotingRegressor | 11.6102 | **7.8038** | 0.1248 |

---

## 9. Key Findings

* **Best model:** XGBoost had the lowest RMSE and highest R² both in cross-validation and on the test set.
* **Ensemble:** The VotingRegressor did not beat XGBoost overall — slightly lower MAE, but higher RMSE and lower R².
* **Limited explained variance:** All models reach R² ≈ 0.15 at best, so most of the variation in weekly hours is not explained by census demographics alone.
* **Influential features:** Sex, income bracket, unknown occupation and household relationship (husband, own child) carried the strongest ANOVA signal.

---

## 10. Repository Structure

```text
machine-learning-predict-hours-per-week/
├── README.md
├── LICENSE                                   <-- MIT
└── ML_Predict_hours_per_week_Project.ipynb   <-- Full analysis: EDA → preparation → models → ensemble
```

---

## 11. Complete Step-by-Step Reproduction Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/alwyn-munatsi/machine-learning-predict-hours-per-week.git
cd machine-learning-predict-hours-per-week
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels xgboost jupyter
```

### Step 3: Run the Notebook
```bash
jupyter notebook ML_Predict_hours_per_week_Project.ipynb
```
Run all cells. The data is downloaded automatically from the DAV-6150 repository, so an internet connection is required.

### Step 4: Review Results
Sections 7 and 8 of the notebook print the cross-validation table, test-set metrics and ensemble comparison shown above. Because every split and model uses `random_state=42`, results should match closely.

---

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE).
