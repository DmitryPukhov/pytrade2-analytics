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

    @staticmethod
    def plot_value_counts(ax, df, col, grouped, name):
        signals = df[col]
        #vc = signals[signals.diff() != 0].value_counts()
        vc = df[col].value_counts() if not grouped else signals[(signals.diff() != 0) & (signals != 0)].value_counts()
        label_map={0:'oom', 1:'buy', -1: 'sell'}
        color_map={'oom':'C0', 'buy': 'C1', 'sell': 'C2'}
        labels = [ label_map[signal] for signal in vc.index.tolist()]
        colors = [color_map[key] for key in labels]
        ax.pie(vc, labels = labels,  autopct= lambda x: '{:.0f}'.format(x*vc.sum()/100), colors = colors)
        tag = 'groups' if grouped else ''
        ax.set_title(f"{name} {col} {tag}")

    @staticmethod
    def plot_signal_counts(df, name = ''):
        #df = df.groupby('close_time').agg('last')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        PlotTool.plot_value_counts(ax1, df, 'signal', grouped = False, name=name)
        PlotTool.plot_value_counts(ax2, df, 'signal', grouped = True, name=name)
        fig.suptitle(f'{name} signal counts from {df.index.min()} to {df.index.max()}')
        plt.show()

    @staticmethod
    def plot_buy_sell(profit_df, name = ''):
        profit_cnt = len(profit_df[(profit_df['signal'] != 0) & (profit_df['profit'] > 0)])
        loss_cnt = len(profit_df[profit_df['profit'] < 0])
        buy_profit_cnt = len(profit_df[(profit_df['signal']== 1) & (profit_df['profit'] >= 0)])
        buy_loss_cnt = len(profit_df[(profit_df['signal']== 1) & (profit_df['profit'] < 0)])
        sell_profit_cnt = len(profit_df[(profit_df['signal']== -1) & (profit_df['profit'] >= 0)])
        sell_loss_cnt = len(profit_df[(profit_df['signal']== -1) & (profit_df['profit'] < 0)])
        #
        # all_data = {'profit': profit_cnt, 'loss': loss_cnt}
        # buy_data = {'profit': buy_profit_cnt, 'loss': buy_loss_cnt}
        # sell_data = {'profit': sell_profit_cnt, 'loss': sell_loss_cnt}

        fig, (ax_all, ax_buy, ax_sell) = plt.subplots(1,3)
        ax_all.pie([profit_cnt, loss_cnt], labels=['profit','loss'], autopct= lambda x: '{:.0f}'.format(x/100*(profit_cnt+loss_cnt)))
        ax_all.set_title('all')
        ax_buy.pie([buy_profit_cnt, buy_loss_cnt], labels=['profit','loss'], autopct= lambda x: '{:.0f}'.format(x/100*(buy_profit_cnt+buy_loss_cnt)))
        ax_buy.set_title('buy')
        ax_sell.pie([sell_profit_cnt, sell_loss_cnt], labels=['profit','loss'], autopct= lambda x: '{:.0f}'.format(x/100*(sell_profit_cnt+sell_loss_cnt)))
        ax_sell.set_title('sell')
        fig.suptitle(f'{name} buy/sell counts')
        plt.show()