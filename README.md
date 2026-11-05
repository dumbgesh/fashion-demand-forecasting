# hm-fashion-demand-forecasting

Time-series forecasting of H&M fashion sales using Python and Prophet.

---

## Forecast Visualisation

### 60-Day Test Window: Actual vs Predicted Sales (Swimwear)
![actual vs predicted sales](assets/actual_vs_predicted.png)

### 30-Day Future Demand Projections Across Target Categories
![30-day future sales forecast](assets/future_forecast.png)

---

## Overview

Effective inventory planning is one of the most critical challenges in fast-fashion retail. Highly seasonal product lines, short product lifecycles, and fluctuating customer demand create significant risks of over-ordering (margin-eroding clearance markdowns) and under-ordering (stockouts and lost revenue).

This project develops an end-to-end time-series forecasting pipeline to predict daily units sold across selected fashion categories. Analyzing 31.78 million customer transactions spanning 734 continuous days, we focus on the top 6 product lines representing **94.05%** of all recorded sales:

1. **Garment Upper body** (39.49% of sales)
2. **Garment Lower body** (22.17% of sales)
3. **Garment Full body** (11.18% of sales)
4. **Swimwear** (8.11% of sales)
5. **Underwear** (8.07% of sales)
6. **Accessories** (5.03% of sales)

A key finding from exploratory analysis is the **Swimwear divergence**: while Swimwear represents less than 3% of catalog designs (3,127 items, ranking 7th), it ranks **4th in actual transaction volume** (2,579,222 units sold). This demonstrates why inventory planning must be driven by demand velocity rather than catalog width.

---

## Dataset

Data is sourced from the Kaggle competition: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

The pipeline uses two core tables:
- `articles.csv` (36.1 MB): Descriptive metadata for 105,542 unique garment items.
- `transactions_train.csv` (3.49 GB): 31,788,324 individual purchase transactions.

**Timeline:** The historical dataset covers continuous daily records from **20 September 2018 to 22 September 2020** (734 days).

> **Important Note:** In compliance with Kaggle competition terms and repository size limits, raw datasets are not redistributed in this repository. Users must download `articles.csv` and `transactions_train.csv` from Kaggle and place them into `data/raw/`.

---

## Technologies Used

- **Python 3.13**
- **DuckDB**: Used for streaming out-of-core SQL joins and aggregations directly from disk, processing 3.49 GB of transactions in ~10 seconds on an 8 GB RAM laptop.
- **Pandas & NumPy**: Data wrangling, grid alignment, and datetime indexing.
- **Facebook Prophet**: Generalized additive model for daily time-series forecasting with weekly and yearly seasonality.
- **Scikit-Learn**: Quantitative model evaluation (MAE and RMSE).
- **Matplotlib**: Clean data visualizations and trend analysis.

---

## Methodology

The project is structured into three self-contained Jupyter notebooks executed sequentially:

1. **Data Understanding & Exploratory Analysis (`notebooks/01_data_understanding.ipynb`)**
   - Assesses storage requirements, schemas, and missing values.
   - Evaluates catalog assortment distribution versus actual consumer transaction volumes.
   - Evaluates macro-level daily transaction trajectories and day-of-week demand patterns.

2. **Data Preparation & Zero-Filling (`notebooks/02_data_preparation.ipynb`)**
   - Streams `transactions_train.csv` and `articles.csv` via DuckDB to avoid memory exhaustion.
   - Builds a complete Cartesian calendar grid across all 734 calendar days for the target categories ($734 \times 6 = 4,404$ records).
   - Zero-fills dates without transactions (`fillna(0)`).
   - Exports the processed modelling dataset to `data/processed/daily_category_sales.csv`.

3. **Prophet Model Training & Evaluation (`notebooks/03_prophet_forecasting.ipynb`)**
   - Implements a strict **chronological train-test split**:
     - **Train period:** 2018-09-20 to 2020-07-24 (674 days, ~22 months).
     - **Hold-out test period:** 2020-07-25 to 2020-09-22 (60 days, ~2 months).
   - Formats data into Prophet conventions (`ds`, `y`) and fits additive models:
     $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
   - Evaluates out-of-sample forecasts on the 60-day test window using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Retrains models on all 734 historical days to generate a 30-day forward-looking forecast (2020-09-23 to 2020-10-22), exported to `data/processed/future_sales_forecast.csv`.

---

## Model Evaluation

Models were evaluated out-of-sample across the final 60 days of the dataset:

| Product Category | Test Mean Actual (Units/Day) | MAE (Units) | RMSE (Units) | MAE Relative Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Underwear** | 3,150.3 | **495.2** | **659.8** | **15.7%** |
| **Garment Lower body** | 8,649.0 | **1,708.8** | **2,188.5** | **19.8%** |
| **Garment Full body** | 4,643.6 | **1,002.5** | **1,338.8** | **21.6%** |
| **Accessories** | 2,015.9 | **453.3** | **560.6** | **22.5%** |
| **Garment Upper body** | 15,608.3 | **3,924.5** | **4,909.3** | **25.1%** |
| **Swimwear** | 2,580.3 | **1,830.8** | **2,092.3** | **71.0%** |

### Evaluation Insights
- **Most predictable category:** `Underwear` achieved the lowest error rate (15.7% MAE). Staple replenishment products maintain consistent demand and clear weekly shopping cycles throughout the year.
- **Moderate categories:** `Garment Lower body` (19.8%), `Full body` (21.6%), and `Accessories` (22.5%) demonstrated predictable seasonal behaviour.
- **Most challenging category:** `Swimwear` exhibited the highest error (71.0% MAE). The 60-day evaluation window (late July through late September) captures the sharp post-summer demand collapse. Without multi-year observations to fine-tune the inflection point, the model under-estimated the steepness of the autumn drop-off.

> **Forecasting Disclaimer:** The 30-day forward projections (`data/processed/future_sales_forecast.csv`) cover 23 September 2020 through 22 October 2020 immediately following the end of the Kaggle historical dataset. They are model projections beyond historical data and do not represent current fashion demand.

---

## Project Structure

```text
hm-fashion-demand-forecasting/
├── assets/
│   ├── actual_vs_predicted.png       # Test window prediction plot (Swimwear)
│   └── future_forecast.png           # 30-day forward projection plot
├── data/
│   ├── raw/                          # Raw Kaggle CSVs (excluded via .gitignore)
│   │   ├── articles.csv
│   │   └── transactions_train.csv
│   └── processed/                    # Processed modelling outputs (excluded via .gitignore)
│       ├── daily_category_sales.csv
│       └── future_sales_forecast.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Exploratory data analysis & sales ranking
│   ├── 02_data_preparation.ipynb     # DuckDB streaming & calendar zero-filling
│   └── 03_prophet_forecasting.ipynb  # Prophet training, evaluation & projections
├── reports/
│   └── data_inspection.md            # Comprehensive project documentation report
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/dumbgesh/fashion-demand-forecasting.git
cd fashion-demand-forecasting
```

### 2. Set Up Environment & Dependencies
Ensure Python 3.10+ is installed:
```bash
pip install pandas numpy matplotlib duckdb prophet scikit-learn
```

### 3. Add Raw Data
Download `articles.csv` and `transactions_train.csv` from the [Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and place them in:
```text
data/raw/articles.csv
data/raw/transactions_train.csv
```

### 4. Execute Notebooks
Run the notebooks in sequential order:
1. `notebooks/01_data_understanding.ipynb`
2. `notebooks/02_data_preparation.ipynb` (generates `data/processed/daily_category_sales.csv`)
3. `notebooks/03_prophet_forecasting.ipynb` (trains models, computes metrics, and generates future forecasts)

---

## Acknowledgements

- **H&M Group** and **Kaggle** for providing the dataset via the *H&M Personalized Fashion Recommendations* competition.
- **Facebook Open Source** for the Prophet forecasting library.
