# fashion demand forecasting & seasonal inventory planning

a college data science project on daily sales forecasting for h&m product categories using prophet.

---

## 1. introduction

in this project we built a daily demand forecasting pipeline for h&m fashion categories using Facebook Prophet.

fashion retail is tricky because demand changes a lot depending on the season and day of the week. if a store stocks too much inventory, items sit on shelves and have to be discounted. if they stock too little, they run out of stock and lose sales. our goal is to predict daily demand so stores can plan their inventory better and avoid overstocking or stockouts.

---

## 2. problem statement

fashion items have short life cycles and strong seasonal demand swings. for example, swimwear sells heavily in summer and barely at all in winter, while staple items like underwear stay relatively constant.

we need to:
- take 31.78 million raw transaction records and aggregate them into daily units sold per product category.
- handle missing dates by filling them with zero sales.
- split the data chronologically into a train set and a test set.
- train a Prophet model for each selected category.
- evaluate predictions using MAE and RMSE on the hold-out test set.
- forecast the next 30 days of future demand to help inventory planning.

---

## 3. dataset

we used data from the kaggle competition: "H&M Personalized Fashion Recommendations".

we downloaded only two files instead of the full dataset:
- `articles.csv` (36.1 mb): metadata for 105,542 unique fashion items. includes columns like `article_id`, `product_group_name`, `department_name`, and `index_group_name`. only 416 items have missing text descriptions; all category columns are 100% complete.
- `transactions_train.csv` (3.49 gb): 31,788,324 individual purchase records from 2018-09-20 to 2020-09-22. contains `t_dat`, `customer_id`, `article_id`, `price`, and `sales_channel_id`. there are zero missing values.

the transactions file spans exactly 734 continuous days with no missing dates in the log.

---

## 4. solution approach

we organized our project into three jupyter notebooks:

1. `notebooks/01_data_understanding.ipynb`: inspect file sizes, missing values, catalog counts, and join transactions to find which categories actually sell the most.
2. `notebooks/02_data_preparation.ipynb`: use DuckDB to aggregate daily transactions per category without crashing our 8 gb ram laptop. build a complete 734-day grid, fill missing days with 0, and export `daily_category_sales.csv`.
3. `notebooks/03_prophet_forecasting.ipynb`: split the data chronologically (first 674 days train, final 60 days test), train Prophet models, evaluate with MAE and RMSE, and forecast the next 30 days.

---

## 5. data analysis

### the 8 gb ram problem
loading a 3.5 gb csv directly with `pd.read_csv()` crashes on an 8 gb laptop because pandas uses too much memory. we solved this by querying the csv directly from disk with DuckDB. the whole join and aggregation took only about 10 seconds and used almost no ram.

### sales volume vs catalog size
we compared how many items each category has in the catalog versus how many units customers actually bought:

| rank | category (`product_group_name`) | units sold | sales % | catalog count | catalog % |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Garment Upper body | 12,552,755 | 39.49% | 42,741 | 40.50% |
| 2 | Garment Lower body | 7,046,054 | 22.17% | 19,812 | 18.77% |
| 3 | Garment Full body | 3,552,470 | 11.18% | 13,292 | 12.59% |
| 4 | Swimwear | 2,579,222 | 8.11% | 3,127 | 2.96% |
| 5 | Underwear | 2,565,858 | 8.07% | 5,490 | 5.20% |
| 6 | Accessories | 1,599,593 | 5.03% | 11,158 | 10.57% |
| 7 | Shoes | 745,521 | 2.35% | 5,283 | 5.01% |
| 8 | Socks & Tights | 685,712 | 2.16% | 2,442 | 2.31% |
| 9 | Nightwear | 348,180 | 1.10% | 1,899 | 1.80% |

### key observations
- **the Swimwear surprise:** swimwear is less than 3% of the catalog (only 3,127 articles), but ranks 4th in actual sales with 2.58 million units. this proves that stocking inventory based only on catalog share is a mistake.
- **sales concentration:** the top 6 categories make up 94.05% of all units sold across the retailer.
- **seasonality:** swimwear has massive spikes in june and july, garment upper body peaks in autumn and winter, while underwear stays flat all year.
- **day of week:** weekends have noticeably higher sales than weekdays, so models need weekly seasonality.

### clean dataset summary
we created a complete grid of 734 days for the top 6 categories ($734 \times 6 = 4,404$ rows) and filled missing dates with 0:

| category | total days | missing values | min units | max units | avg daily units |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Garment Upper body | 734 | 0 | 5,277 | 104,168 | 17,101.8 |
| Garment Lower body | 734 | 0 | 2,716 | 43,045 | 9,599.5 |
| Garment Full body | 734 | 0 | 1,086 | 22,534 | 4,839.9 |
| Swimwear | 734 | 0 | 169 | 34,057 | 3,513.9 |
| Underwear | 734 | 0 | 1,228 | 17,137 | 3,495.7 |
| Accessories | 734 | 0 | 543 | 10,352 | 2,179.3 |

the dataset was saved to `data/processed/daily_category_sales.csv`.

---

## 6. forecasting model & train-test split

we used Facebook Prophet for time-series forecasting.

Prophet breaks down each daily sales series as:
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
where:
- $g(t)$ is the trend (overall growth or decline)
- $s(t)$ is seasonality (weekly and yearly cyclic patterns using Fourier terms)
- $h(t)$ is holiday effects
- $\epsilon_t$ is random error

Prophet requires two input columns:
- `ds`: the date
- `y`: the value to predict (`units_sold`)

### train-test split
we used a chronological split instead of random splitting to prevent data leakage:
- **train set:** 2018-09-20 to 2020-07-24 (674 days, ~22 months)
- **test set:** 2020-07-25 to 2020-09-22 (60 days, ~2 months)

this gave the models almost two full annual cycles to learn seasonal patterns, while testing on a realistic 60-day forecasting window.

---

## 7. evaluation results

we evaluated predictions against actual sales on the 60-day test set using MAE and RMSE:
- **MAE (Mean Absolute Error):** average daily error in units.
- **RMSE (Root Mean Squared Error):** penalizes bigger forecasting misses.

here are the actual results from our models:

| category | avg daily sales | MAE (units) | RMSE (units) | MAE % |
| :--- | :---: | :---: | :---: | :---: |
| **Underwear** | 3,150.3 | **495.2** | **659.8** | **15.7%** |
| **Garment Lower body** | 8,649.0 | 1,708.8 | 2,188.5 | 19.8% |
| **Garment Full body** | 4,643.6 | 1,002.5 | 1,338.8 | 21.6% |
| **Accessories** | 2,015.9 | 453.3 | 560.6 | 22.5% |
| **Garment Upper body** | 15,608.3 | 3,924.5 | 4,909.3 | 25.1% |
| **Swimwear** | 2,580.3 | 1,830.8 | 2,092.3 | **71.0%** |

### which categories performed better and why?
- **best performer: Underwear (15.7% error)**. demand is very steady and non-cyclical. people replace underwear consistently all year round, so Prophet easily captured the baseline and weekly patterns.
- **good performers: Garment Lower body (19.8%), Full body (21.6%), and Accessories (22.5%)**. these categories have relatively smooth demand curves with regular weekly peaks.
- **moderate performer: Garment Upper body (25.1% error)**. volume is very high (over 15,000 units/day) with large seasonal swings between summer shirts and winter jackets/sweaters.
- **hardest category: Swimwear (71.0% error)**. swimwear demand collapses very fast during august and september as summer ends. because we only had 674 days of training data, the model struggled to predict exactly how fast demand would drop off after the july peak.

actual vs predicted plots for all 6 categories are rendered in `notebooks/03_prophet_forecasting.ipynb`.

---

## 8. 30-day future forecasts

after evaluating on the test set, we retrained each model on all 734 days of data and generated forecasts for the next 30 days (from 2020-09-23 to 2020-10-22).

these future predictions were saved to `data/processed/future_sales_forecast.csv` (180 rows: 30 days $\times$ 6 categories).

*note: these are model projections into the future beyond the historical dataset, not verified future sales.*

visualizations of the 30-day forecast across all categories are displayed in `notebooks/03_prophet_forecasting.ipynb`.

---

## 9. conclusion & inventory planning recommendations

what we learned and how this helps inventory planning:

1. **use transaction volume, not catalog size:** stocking decisions should be guided by sales velocity rather than catalog width. swimwear only accounts for 2.96% of the catalog but represents 8.11% of total sales.
2. **category-specific inventory strategies:**
   - **stable categories (Underwear, Lower body):** low forecasting error (~16–20%) means stores can keep lean safety stock and use regular automated weekly replenishment.
   - **seasonal categories (Swimwear):** high error (~71%) during seasonal transitions means retailers must start ramping down inventory aggressively in late july to avoid holding dead stock through winter.
   - **high volume categories (Upper body):** large unit volumes mean small percentage errors translate into thousands of units. safety buffers should be increased ahead of autumn/winter stock changeovers.
3. **limitations:**
   - the dataset has 2 years of history, which is the bare minimum for estimating annual seasonality. with 3 or 4 years of data, yearly seasonality would be much more reliable.
   - external factors like temperature anomalies, local weather, and specific marketing promotions were not available in the dataset and could further improve accuracy.
