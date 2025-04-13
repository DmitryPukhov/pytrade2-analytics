import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from pytrade2.pytrade2.features.LowHighTargets import LowHighTargets
from pytrade2.pytrade2.features.MultiIndiFeatures import MultiIndiFeatures


class DataTool:

    @staticmethod
    def read_last_candles(ticker, data_dir, days=1):
        """ Read 1min candles from data_dir for given days """

        file_paths = sorted(
            [f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith(f"{ticker}_candles_1min.csv")])[-days:]
        data = pd.concat([pd.read_csv(f, parse_dates=["open_time", "close_time"]) for f in file_paths])
        data.set_index("close_time", drop=False, inplace=True)
        return data

    @staticmethod
    def candles_by_periods_of(candles_1min: pd.DataFrame, periods: list[str]) -> dict[str, pd.DataFrame]:
        """ Create candles_by_periods """
        out = {}
        for period in periods:
            out[period] = candles_1min.resample(period).agg({'open_time': 'first',
                                                             'close_time': 'last',
                                                             'open': 'first',
                                                             'high': 'max',
                                                             'low': 'min',
                                                             'close': 'last',
                                                             'vol': 'max'
                                                             })
        return out

    @staticmethod
    def train_test_split(x, y, test_days=14):
        """ In addition to train_test_split, create pipeline for reverse transform in future"""
        test_index = max(x.index) - pd.Timedelta(days=test_days)
        is_test = (x.index >= test_index)
        is_train = (x.index < test_index)

        x_train, y_train, x_test, y_test = x[is_train], y[is_train], \
            x[is_test], y[is_test]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def create_pipe(x, y) -> (Pipeline, Pipeline):
        """ Create feature and target pipelines to use for transform and inverse transform """

        time_cols = [col for col in x.columns if col.startswith("time")]
        float_cols = list(set(x.columns) - set(time_cols))

        x_pipe = Pipeline(
            [("xscaler", ColumnTransformer([("xrs", StandardScaler(), float_cols)], remainder="passthrough")),
             ("xmms", MaxAbsScaler())])
        x_pipe.fit(x)

        y_pipe = Pipeline(
            [("yrs", StandardScaler()),
             ("ymms", MaxAbsScaler())])
        y_pipe.fit(y)
        return x_pipe, y_pipe