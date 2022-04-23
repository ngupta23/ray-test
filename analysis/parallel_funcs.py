import pandas as pd
from pycaret.time_series import TSForecastingExperiment


def forecast_single(data: pd.DataFrame) -> pd.DataFrame:
    """Forecasts a single item and returns the predictions

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing a single item and its historical values

    Returns
    -------
    pd.DataFrame
        Predictions for the item
    """

    item = None
    if "Item" in data.columns:
        item = data["Item"].unique()
        assert len(item) == 1
        item = item[0]
        print(f"Item: {item} | History Available: {len(data)} months")
    else:
        print("Item column not found in dataframe")

    data["YYYYMM"] = pd.PeriodIndex(data["YYYYMM"], freq="M")
    data = data.sort_values("YYYYMM")
    data.set_index("YYYYMM", inplace=True)

    FH = 12
    FOLDS = 3
    SP = None

    # Overrides for Items with minimal data
    if len(data) <= 5 * FH:
        dates = data.index.sort_values()
        future_index = [dates[-1] + i for i in range(1, FH + 1)]
        preds = pd.DataFrame(index=future_index)
        preds["y_pred"] = data["Sales"].mean()
    else:
        exp = TSForecastingExperiment()
        exp.setup(
            data=data["Sales"],
            seasonal_period=SP,
            fh=FH,
            fold=FOLDS,
            n_jobs=1,
            session_id=42,
        )
        include = ["arima"]
        best = exp.compare_models(include=include)
        metrics = exp.pull()
        print(exp.check_stats(test="summary"))
        print(f"Item: {item} | Best: {best} | CV Metrics\n{metrics}")
        final = exp.finalize_model(best)
        preds = exp.predict_model(final)

    preds["y_pred"] = preds["y_pred"].clip(lower=0).round().astype(int)

    # When using with ray, item will not be in index since we are not using
    # pandas groupby. Hence, adding explicitly here.
    preds["Item"] = item
    return preds
