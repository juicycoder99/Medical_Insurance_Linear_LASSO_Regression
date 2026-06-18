# Programming Languages for Data Analysis (CS504) — Assignment 3

Coursework for **Programming Languages for Data Analysis (CS504)**, Department of Computer Science,
Bishop's University.

## Predicting medical insurance cost with linear and LASSO regression

Building regression models to predict individual medical charges from age, sex, BMI, number of
children, smoker status and region. The dataset is the well-known
[insurance.csv](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv)
(1,338 records).

The solution is in [`Assignment_3.ipynb`](Assignment_3.ipynb).

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
jupyter notebook Assignment_3.ipynb
```

## Files

| File | Description |
|------|-------------|
| `Assignment_3.ipynb` | Full solution |
| `insurance.csv` | Dataset (stedy/Machine-Learning-with-R-datasets) |
