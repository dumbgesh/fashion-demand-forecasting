# fashion demand forecasting & seasonal inventory planning

a college data science project forecasting daily sales for h&m fashion product categories using facebook prophet.

## about the project

fashion retail inventory is hard to plan because demand changes across seasons and days of the week. in this project, we analyze 31.78 million transaction records from h&m, prepare daily sales time series for the top 6 product categories, and forecast sales using prophet.

## dataset

dataset source: [kaggle h&m personalized fashion recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

we use two files from the competition:
- `articles.csv`: metadata for 105,542 items.
- `transactions_train.csv`: 31,788,324 purchase events across 734 continuous days (2018-09-20 to 2020-09-22).

**note:** raw data is not included in this repository due to kaggle competition rules and file size (~3.5 gb). download `articles.csv` and `transactions_train.csv` directly from kaggle and place them in `data/raw/`.

## tools & libraries

- python 3.13
- duckdb (for querying and aggregating 3.5 gb csv on an 8 gb ram laptop in ~10 seconds)
- pandas & numpy
- matplotlib
- prophet (time-series forecasting)
- scikit-learn (for mae and rmse metrics)

## notebook execution order

run the notebooks in this order:

1. `notebooks/01_data_understanding.ipynb`
   - inspects file sizes, column types, and missing values.
   - checks catalog count vs actual sales share (uncovers that swimwear ranks 4th in sales despite being <3% of the catalog).
   - visualizes daily sales trends and day-of-week patterns.

2. `notebooks/02_data_preparation.ipynb`
   - streams and aggregates transactions with duckdb.
   - builds a complete 734-day grid for the top 6 categories.
   - zero-fills any missing dates.
   - exports the clean modeling data to `data/processed/daily_category_sales.csv`.

3. `notebooks/03_prophet_forecasting.ipynb`
   - chronological train-test split: first 674 days for training, last 60 days for testing.
   - trains prophet models for each category.
   - evaluates forecasts using mae and rmse.
   - retrains on all historical data and forecasts the next 30 days (saved to `data/processed/future_sales_forecast.csv`).

## model & evaluation metrics

we use prophet to decompose daily sales into:
$$y(t) = \text{trend} + \text{seasonality} + \text{holidays} + \text{error}$$

we evaluate on the 60-day test window with:
- **mae (mean absolute error):** average daily error in units sold.
- **rmse (root mean squared error):** penalizes bigger forecasting misses.

results summary:
- underwear was the easiest to forecast (15.7% mae error) due to steady baseline demand.
- swimwear was the hardest (71.0% mae error) due to the steep post-summer demand drop.

## project report

a detailed student report is available in `reports/data_inspection.md`.
