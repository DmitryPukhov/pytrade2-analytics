from datetime import timedelta,date, datetime
import os
import glob


import pandas as pd
import plotly as py
from plotly import graph_objects as go
import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt

class PlotTool:

    @staticmethod
    def plot_candles(df: pd.DataFrame, period: str)->pd.DataFrame:
        """
        Candles chart resampled to period
        """
        ticker = df.iloc[0]["ticker"]
        df=df.resample(period).agg({"open": "first", "high": "max", "low": "min", "close": "last", "open_time": "first", "close_time": "last"})
        fig = go.Figure(data=[ \
            go.Candlestick( \
                x=df.index, \
                open=df['close'].shift(1), \
                high=df['high'], \
                low=df['low'], \
                close=df['close'])
        ])


        fig.update_layout(title_text=f"{ticker} {period} candles")
        fig.show()