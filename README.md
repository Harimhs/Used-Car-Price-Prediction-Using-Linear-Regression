***

# Used Car Price Prediction: End-to-End Regression Project


***

## Introduction

This project demonstrates the process of building a robust machine learning regression model to predict used car prices. The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, model implementation from scratch, hyperparameter tuning using inbuilt libraries, and model evaluation.

The journey emphasizes not just obtaining high accuracy, but deeply understanding the methods used, reflecting on challenges and solutions, and preparing the workflow for production usage.

***

## Data Preprocessing \& Feature Engineering

We began with reading the raw dataset and handling data quality issues:

- Dropped irrelevant columns like `"accidents_reported"` early to streamline data.
- Checked for missing data, especially in `"service_history"`, filling missing values with `"unknown"`.
- Created new features such as `car_age` (2025 minus manufacture year), `car_efficiency`, and `price_per_year` to better capture domain insights.

Snippet – filling missing values and feature engineering:

```python
df["service_history"] = df["service_history"].fillna("unknown")
df["car_age"] = 2025 - df["make_year"]
efficiency = (df["mileage_kmpl"] > 15) & (df["service_history"] == "Full")
df = df.assign(car_efficiency=efficiency, price_per_year=df["price_usd"] / df["car_age"])
```

***

## Encoding \& Cleaning

- Converted categorical features into numerical ones, using binary encoding for features with two unique values and one-hot encoding for others.
- Applied variance threshold filtering to remove low-variance features.
- Removed highly correlated features to handle multicollinearity (features with correlation > 0.8).

Snippet – variance threshold \& multicollinearity removal:

```python
def variance_finding(X, threshold):
    variance = np.var(X, axis=0)
    worth_cols = [i for i, v in enumerate(variance) if v > threshold]
    return X.iloc[:, worth_cols]

X_variance = variance_finding(X, 0.05)
corr_matrix = X_variance.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
X_clean = X_variance.drop(to_drop, axis=1)
```

***

## Train-Test Split \& Feature Selection

- Used a manual implementation of train-test split to ensure reproducibility.
- Employed a post-split correlation filter, keeping features with absolute correlation > 0.1 on training data only.
- Included certain domain knowledge features (all brand dummy variables plus car efficiency), acknowledging their weak individual correlations but domain importance.

Snippet – correlation filter and domain knowledge feature inclusion:

```python
train_corr = X_train.corrwith(y_train).abs()
to_keep = train_corr[train_corr > 0.1].index.tolist()

brand_cols = [col for col in X_train.columns if "brand_" in col]
border = ['car_efficiency'] if 'car_efficiency' in X_train.columns else []
final_features = to_keep + brand_cols + border

X_corr_train = X_train[final_features]
X_corr_test = X_test[final_features]
```

***

## Scaling

- Written a manual z-score standardization method to deeply understand the statistics.
- Scaling parameters (mean and std) were calculated on training data only and applied to test data to avoid data leakage.

Snippet – manual scaling function:

```python
def fit_scale(df):
    df = df.copy().astype(float)
    mu_sigma = {}
    for col in df.columns:
        mu = df[col].mean()
        sigma = df[col].std()
        mu_sigma[col] = {"mu": mu, "sigma": sigma}
        df[col] = (df[col] - mu) / sigma
    return df, mu_sigma
```

***

## Modeling

- First built Linear Regression manually using batch gradient descent, seeing strong alignment between manual and sklearn inbuilt model metrics: both yielded RMSE ~996 and R² ~0.87.
- Then applied hyperparameter tuning with GridSearchCV over Lasso alpha, finding an optimal alpha ~3.79 with cross-validated R² ~0.875.
- Final models showed incredibly close performance.

Snippet – manual model training loop excerpt:

```python
for i in range(self.epoch):
    Y = X @ self.weights.T + self.bias
    error = y - Y
    cost = np.mean(error ** 2)
    dw = (-2 / n_samples) * (X.T @ error)
    db = (-2 / n_samples) * (np.sum(error))
    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
```

Comparison of final metrics:


| Metric | Manual Model | Inbuilt Lasso | Difference |
| :-- | :-- | :-- | :-- |
| MSE | 992,545 | 991,833 | 712 |
| RMSE | 996.27 | 995.91 | 0.36 |
| MAE | 799.28 | 798.88 | 0.40 |
| R² | 0.8694 | 0.8695 | -0.0001 |

***

## Cross-Validation \& Hyperparameter Tuning

- Used KFold 10 splits and GridSearchCV over logarithmic alpha range.
- Validated model performance robustly and selected best hyperparameters with confidence.
- Discussed theoretical basis of CV and tuning, emphasizing importance in preventing overfitting and ensuring model generalizability.

***

## Conclusion \& Lessons

- Manual implementation deepened understanding of model training mechanics.
- Careful preprocessing, feature engineering, and domain knowledge were critical for model success.
- Inbuilt tools efficiently tuned the model for best predictive performance with minimal gap to manual efforts.
- The hybrid approach is recommended: build and understand models yourself but use tools intelligently in production.
- Documentation and code savings ensure reproducibility and readiness for deployment or sharing.

***
