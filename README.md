# Stock Price Prediction Project

This project focuses on predicting stock prices using various machine learning and deep learning models, including traditional methods like ARIMA and advanced architectures like LSTM and Transformers. The dataset used is Apple stock prices, with features such as `Close`, `Volume`, `High`, and `Low`. The project emphasizes time series analysis, lag selection, and model interpretability.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Workflow](#workflow)
- [Models Implemented](#models-implemented)
- [Performance Evaluation](#performance-evaluation)
- [Interpretability](#interpretability)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

---

## **Project Overview**

Stock price prediction is a challenging task due to the inherent volatility and complex dependencies in financial markets. This project leverages historical stock data and advanced machine learning techniques to forecast future prices. The focus is on understanding patterns and improving prediction accuracy using state-of-the-art models.

---

## **Objectives**
- Perform **time series analysis** to identify patterns and trends.
- Explore **lag analysis** to find optimal time steps for modeling.
- Compare traditional models (e.g., ARIMA) with deep learning models (LSTM, Transformers).
- Enhance model interpretability using SHAP and LIME.
- Provide robust and professional-grade code for deployment.

---

## **Dataset**

The dataset includes Apple stock prices with the following features:
- `Date`: Date of the record.
- `Close`: Closing price of the stock.
- `High`: Highest price of the day.
- `Low`: Lowest price of the day.
- `Volume`: Number of shares traded.

The dataset is preprocessed to handle missing values, normalize features, and create lagged features.

---

## **Key Features**
- **Lag Analysis**: Autocorrelation (ACF) and Partial Autocorrelation (PACF) used to determine significant lags.
- **Feature Engineering**: Created lagged and rolling features to capture temporal dependencies.
- **Modeling**: Experimented with ARIMA, LSTM, and Transformer models for predictions.
- **Interpretability**: Utilized SHAP and LIME to explain predictions.
- **Visualization**: Extensive use of Matplotlib, Plotly, and Seaborn for data and model insights.

---

## **Workflow**
1. **Exploratory Data Analysis (EDA):**
   - Visualize trends, seasonality, and anomalies.
   - Perform stationarity tests (ADF Test).

2. **Preprocessing:**
   - Handle missing data and normalize features.
   - Create lagged features and time steps.

3. **Modeling:**
   - Train ARIMA for baseline predictions.
   - Develop LSTM for capturing long-term dependencies.
   - Experiment with Transformer models for enhanced predictions.

4. **Evaluation:**
   - Compare models using metrics like RMSE and MAPE.

5. **Interpretability:**
   - Use SHAP and LIME to interpret model decisions.

---

## **Models Implemented**

### **1. ARIMA**
- Captures linear relationships in time series.
- Requires stationarity in the data.

### **2. LSTM**
- A type of recurrent neural network (RNN).
- Designed to capture long-term dependencies in time series.

### **3. Transformer**
- State-of-the-art deep learning model for sequence data.
- Leverages attention mechanisms to model dependencies.

---

## **Performance Evaluation**
- **Metrics Used:**
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

- **Results:**
  - LSTM outperformed ARIMA in capturing complex dependencies.
  - Transformer provided the most accurate predictions, especially for long horizons.

---

## **Interpretability**

### **SHAP (SHapley Additive exPlanations):**
- Provided global and local insights into feature contributions.
- Visualized the impact of `Close`, `Volume`, and lagged features on predictions.

### **LIME (Local Interpretable Model-agnostic Explanations):**
- Offered quick explanations for specific predictions.

---

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:**
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn, Plotly
  - Modeling: Scikit-learn, Statsmodels, TensorFlow, PyTorch
  - Interpretability: SHAP, LIME

---

## **Model Evaluation**
- LSTM Model Evaluation Results:
- RMSE: 4.0859
- MAPE: 3.95%
- R-Squared: 0.9937


## **Future Work**
- Implement **ensemble models** for better performance.
- Incorporate **macroeconomic indicators** (e.g., interest rates, GDP).
- Optimize Transformer models for real-time predictions.
- Deploy the model as a web app using Flask or FastAPI.

---

## **Usage**
To run the project locally:
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd stock_price_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook
   ```

4. Explore the code and visualizations for insights.

---