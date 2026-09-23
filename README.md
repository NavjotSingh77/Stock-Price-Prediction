# Stock Price Prediction using Machine Learning

A machine learning project for predicting the closing price of **Stock_5** using the closing prices of four other stocks (**Stock_1–Stock_4**).

The notebook performs data cleaning, outlier analysis, normalization using Z-scores, correlation analysis, visualization, and regression-based stock price prediction using **Linear Regression** and **Decision Tree Regression**.

## Project Overview

The objective of this project is to investigate whether the prices of multiple stocks can be used as input variables to predict the price of another stock.

### Prediction Setup

- **Target variable:** `Stock_5`
- **Input features:** `Stock_1`, `Stock_2`, `Stock_3`, `Stock_4`
- **Models used:**
  - Linear Regression
  - Decision Tree Regressor

## Workflow

The notebook follows this pipeline:

1. Load the stock dataset using Pandas.
2. Inspect the dataset shape and columns.
3. Remove duplicate rows.
4. Visualize potential outliers using box plots.
5. Calculate Z-scores for the five stock columns.
6. Rename the time column to `Time`.
7. Sort observations by time.
8. Check for missing values and duplicate rows.
9. Calculate the correlation matrix between the five stocks.
10. Visualize stock price time series.
11. Define `Stock_5` as the prediction target.
12. Split the data into training and testing sets.
13. Train a Linear Regression model.
14. Evaluate predictions using MAE, MSE, RMSE, and R².
15. Train a Decision Tree Regressor.
16. Compare actual and predicted stock prices visually.

## Dataset Structure

The notebook expects a CSV file named:

```text
stock_data.csv
```

The dataset is expected to contain the following columns:

```text
Unnamed: 0
Stock_1
Stock_2
Stock_3
Stock_4
Stock_5
```

`Unnamed: 0` is treated as the time variable and renamed to `Time`.

> The notebook itself does not provide information about the original source, market, ticker symbols, or exact meaning of the five stock columns. Add those details here if you have them.

## Models

### 1. Linear Regression

Linear Regression is used as the baseline model.

The model learns the relationship:

```text
Stock_5 = f(Stock_1, Stock_2, Stock_3, Stock_4)
```

The implementation uses:

```python
LinearRegression()
```

### 2. Decision Tree Regression

A Decision Tree Regressor is also trained with the following configuration:

```python
DecisionTreeRegressor(
    max_depth=12,
    random_state=42,
    splitter='random',
    criterion='absolute_error'
)
```

The Decision Tree is used to capture non-linear relationships between the input stocks and `Stock_5`.

## Evaluation Metrics

The notebook calculates the following regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Lower MAE is better.**

### MSE — Mean Squared Error

Penalizes larger prediction errors more heavily.

**Lower MSE is better.**

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same units as the target variable.

**Lower RMSE is better.**

### R² — R-squared

Measures how much of the variation in the target is explained by the model.

A value closer to **1** generally indicates a better fit on the evaluated data.

## Visualizations

The notebook includes:

- Box plots for `Stock_1` through `Stock_5`
- Correlation heatmap
- Time-series plot of all five stocks
- Actual vs. predicted values for Linear Regression
- Actual vs. predicted values for Linear Regression and Decision Tree

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Jupyter Notebook

## Installation

Clone the repository and install the required packages:

```bash
git clone <your-repository-url>
cd <your-repository-folder>

pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
model.ipynb
```

Make sure `stock_data.csv` is available at the path expected by the notebook, or update the CSV path in the data-loading cell.

## Project Structure

```text
Stock-Price-Prediction/
│
├── model.ipynb
├── stock_data.csv
└── README.md
```

## Important Notes and Limitations

This project is an experimental machine learning study and should **not** be interpreted as a reliable financial trading or investment system.

### Time-Series Validation

The current notebook uses:

```python
train_test_split(x, y, test_size=0.3, random_state=None)
```

For real stock-price forecasting, a chronological train/test split or time-series cross-validation would generally be more appropriate because randomly mixing past and future observations can introduce data leakage.

### Prediction Target

The current implementation predicts only:

```text
Stock_5
```

using:

```text
Stock_1, Stock_2, Stock_3, Stock_4
```

### Decision Tree RMSE Calculation

In the current notebook, the Decision Tree section contains:

```python
rmse_dt = np.sqrt(mse)
```

This uses the Linear Regression MSE variable rather than the Decision Tree MSE. The intended calculation should be based on `mse_dt`:

```python
rmse_dt = np.sqrt(mse_dt)
```

This should be corrected before using the Decision Tree RMSE as a reported result.

## Future Improvements

Possible improvements include:

- Use chronological train/test splitting.
- Use `TimeSeriesSplit` for cross-validation.
- Add lag features such as previous-day prices.
- Add daily returns and percentage changes.
- Add moving averages and rolling statistics.
- Compare Random Forest and Gradient Boosting models.
- Compare XGBoost/LightGBM with appropriate validation.
- Perform hyperparameter tuning.
- Evaluate predictions over different forecasting horizons.
- Add financial features such as volume and volatility if available.
- Save the trained model using `joblib` or `pickle`.
- Build a prediction API using Flask or FastAPI.
- Create an interactive dashboard for predictions and model evaluation.

## Disclaimer

This project is intended for educational and research purposes. Stock prices are affected by many factors that are not represented in this dataset, and model performance on historical data does not guarantee future performance.

## Author

**Navjot Singh**

Machine Learning / Data Science Enthusiast

GitHub: `navjotsinghgit`
