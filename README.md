# Predicting Medical Insurance Costs with Linear and LASSO Regression

Building regression models to predict individual medical charges from age, sex, BMI, number of
children, smoker status, and region. The dataset is the well-known
[insurance.csv](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv)
(1,338 records).

The full implementation is in [`insurance_cost_regression.ipynb`](insurance_cost_regression.ipynb).

## What the notebook does

1. Load the data and inspect it.
2. Plot `charges` against `BMI` and against `age`.
3. Label-encode the categorical variables (`sex`, `smoker`, `region`).
4. Scale every variable to `[0, 1]` with `MinMaxScaler`.
5. Split into 70% training / 30% testing.
6. Show the correlation matrix on the training set and find the three inputs most correlated with
   `charges` (**smoker, age, BMI**).
7. Train a linear regression on those three variables and report training and testing MSE.
8. Train a LASSO model, then use cross-validation (`LassoCV`) to find the best regularisation
   strength and the most important variable (**smoker**).
9. Plot how the LASSO coefficients shrink as lambda increases, and evaluate the best model on the
   test set (MSE ≈ 0.0086, R² ≈ 0.77).

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook insurance_cost_regression.ipynb
```

## Files

| File | Description |
|------|-------------|
| `insurance_cost_regression.ipynb` | Full implementation and analysis |
| `insurance.csv` | Dataset (stedy/Machine-Learning-with-R-datasets) |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
