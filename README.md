# Mouna_loa-Co2-prediction

This project analyzes atmospheric CO2 levels from the Mouna Loa Observatory and forecasts future concentrations using a SARIMAX time series model.

## 🚀 Live Demo
Access the deployed Streamlit application here:
**[https://mounaloa-co2-prediction-7f8bdn3hpbgxdu](https://mounaloa-co2-prediction-7f8bdn3hpbgxdu)**

## 📊 Features
- **Data Visualization**: Interactive plots of historical CO2 data.
- **Seasonality Analysis**: Decomposes the time series into trend, seasonal, and residual components.
- **Forecasting**: Predicts future CO2 levels using a trained SARIMAX(1,1,1)x(1,0,1,12) model.
- **Model Evaluation**: Displays MAE and RMSE metrics for model performance.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Krishna417211/Mouna_loa-Co2-prediction.git
    cd Mouna_loa-Co2-prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
- `app.py`: The main Streamlit application.
- `Mouna_Loa.ipynb`: Jupyter notebook containing the exploratory data analysis and model training logic.
- `Project_Report.md`: Detailed report on the project's objective, methodology, and results.
- `requirements.txt`: List of Python dependencies.

## 📝 Dataset
The application uses `archive.csv` which contains monthly CO2 measurements. Ensure this file is present in the root directory when running locally.

https://mounaloa-co2-prediction-7f8bdn3hpbgxdu
live site
