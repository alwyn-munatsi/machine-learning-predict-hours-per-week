# Project Title: THE QUEST TO PREDICT LABOR: A DATA-DRIVEN LOOK AT WEEKLY WORK HOURS

## Project Members
- Alwyn Munatsi
- Lucia Shumba
- Chidochashe Makanga
- Bekithemba Nkomo

## Project Supervisor
- James Topor

## Overview
This project explores the factors that influence weekly labor supply (hours worked per week) among U.S. adults using the Adult (Census Income) dataset from the UCI Machine Learning Repository. We aim to identify key demographic, socioeconomic, and occupational variables that best predict weekly work hours and evaluate the performance of various regression-based machine learning models, including an ensemble approach.

## Dataset
- **Name:** UCI Adult (Census Income) dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Description:** Consists of 48,842 observations from the 1994 U.S. Census, with 15 variables covering demographic, educational, occupational, and economic characteristics.
- **Target Variable:** `hours-per-week` (continuous numeric)

## Research Questions
1.  Which demographic, educational, and occupational attributes most strongly influence weekly labor supply?
2.  Can machine learning regression models accurately predict weekly hours worked using demographic and socioeconomic features?
3.  Does an ensemble regression model outperform individual regression models when predicting weekly hours worked?

## Methodology
1.  **Data Ingestion:** Loading the dataset from the UCI repository.
2.  **Exploratory Data Analysis (EDA):** Univariate and bivariate analysis to understand data distributions, relationships, and identify data quality issues.
3.  **Data Cleaning and Preprocessing:** Handling missing values, dropping irrelevant or redundant columns (`fnlwgt`, `education`, `marital_status`), converting `capital_gain` and `capital_loss` to binary features, grouping infrequent categories in `native_country`, and applying one-hot encoding to categorical variables.
4.  **Feature Selection:** Using `SelectKBest` with ANOVA F-value to identify the most significant features.
5.  **Model Training:** Implementing and tuning five regression models:
    *   Negative Binomial Regression
    *   K-Nearest Neighbors (KNN) Regressor
    *   Support Vector Regression (SVR)
    *   Decision Tree Regressor
    *   XGBoost Regressor
6.  **Model Evaluation:** Assessing model performance using RMSE, MAE, and R² with five-fold cross-validation.
7.  **Ensemble Integration:** Constructing a `VotingRegressor` combining selected base learners.
8.  **Interpretation of Results:** Analyzing model insights and comparing ensemble performance against individual models.

## Key Findings
-   **Significant Predictors:** `age`, `education_num`, `workclass`, `occupation`, `relationship`, `sex`, and `income` were found to be statistically significant predictors of `hours_per_week`.
-   **Model Performance:** The **XGBoost Regressor** emerged as the best individual model with the lowest RMSE (11.4715) and highest R² (0.1456) on the test set. All models showed relatively low R², indicating that a substantial portion of variance in `hours_per_week` remains unexplained by the current features.
-   **Ensemble Performance:** The `VotingRegressor` did not universally outperform the best individual model. While it achieved a slightly lower MAE (7.8038), its RMSE (11.6102) was marginally higher and R² (0.1248) noticeably lower than XGBoost.

## Installation
To run this project, you will need Python 3.x and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

## Usage
1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook ML_Predict_hours_per_week_Project.ipynb
    ```
3.  Run all cells in the notebook to reproduce the analysis, model training, and evaluation.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
