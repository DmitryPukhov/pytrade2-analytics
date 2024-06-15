import pandas as pd

from pytrade2.pytrade2.strategy.features.LowHighTargets import LowHighTargets
from pytrade2.pytrade2.strategy.features.MultiIndiFeatures import MultiIndiFeatures
from pytrade2.pytrade2.strategy.signal.SignalByFutLowHigh import SignalByFutLowHigh
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from keras import Sequential, Input
from keras.layers import Dense, Dropout
from sklearn.linear_model import LinearRegression

class LowHighRegressionTool:

    @staticmethod
    def build_features_targets(candles_by_period: dict[str, pd.DataFrame], target_period: str, features_params=None):
        features = MultiIndiFeatures.multi_indi_features(candles_by_period, features_params)
        targets = LowHighTargets.fut_lohi(candles_by_period[target_period], target_period)
        features = features.dropna()
        targets = targets.dropna()
        features = features[features.index.isin(targets.index)]
        targets = targets[targets.index.isin(features.index)]
        return features, targets

    @staticmethod
    def calc_signal_ext(candles, y_test, y_pred,
                        profit_loss_ratio,
                        stop_loss_min_coeff,
                        stop_loss_max_coeff,
                        stop_loss_add_ratio,
                        take_profit_min_coeff,
                        take_profit_max_coeff,
                        comission_pct,
                        price_precision
                        ):
        """ Using predicted values, create dataframe with signal, stoploss, takeprofit """
        # Combine test with pred
        fut_df = pd.DataFrame(index=y_pred.index)
        fut_df['close'] = candles['close']
        fut_df['fut_low'] = candles['close'] + y_test['fut_low_diff']
        fut_df['fut_high'] = candles['close'] + y_test['fut_high_diff']
        fut_df['fut_low_pred'] = candles['close'] + y_pred['fut_low_diff']
        fut_df['fut_high_pred'] = candles['close'] + y_pred['fut_high_diff']

        # Signal calculator from pytrade2
        signal_calc = SignalByFutLowHigh(profit_loss_ratio=profit_loss_ratio,
                                         stop_loss_min_coeff=stop_loss_min_coeff,
                                         stop_loss_max_coeff=stop_loss_max_coeff,
                                         stop_loss_add_ratio=stop_loss_add_ratio,
                                         take_profit_min_coeff=take_profit_min_coeff,
                                         take_profit_max_coeff=take_profit_max_coeff,
                                         comission_pct=comission_pct,
                                         price_presision=price_precision)

        # Create df: signal, stoploss, take profit
        signal_df = fut_df[['close', 'fut_low_pred', 'fut_high_pred']] \
            .apply(lambda row: signal_calc.calc_signal(row[0], row[1], row[2]), axis=1, result_type='expand')
        signal_df.columns = ['signal', 'sl', 'tp']
        signal_df = pd.concat([signal_df, fut_df], axis=1)
        signal_df = LowHighRegressionTool.with_profit(signal_df, comission=comission_pct * 0.01)
        signal_df = LowHighRegressionTool.with_drawdown(signal_df)
        return pd.concat([signal_df, fut_df], axis=1)

    @staticmethod
    def with_profit(signal: pd.DataFrame, comission: float):
        """ Metric: profit it trade by predicted values """

        # Buy and price keeps above stop loss and goes over tp
        is_buy_profit = (signal['signal'] == 1) & (signal['fut_low'] > signal['sl']) & (
                signal['fut_high'] >= signal['tp'])
        is_buy_loss = (signal['signal'] == 1) & (~is_buy_profit)

        # Sell and price keeps below stop loss and goes below tp
        is_sell_profit = (signal['signal'] == -1) & (signal['fut_high'] < signal['sl']) & (
                signal['fut_low'] <= signal['tp'])
        is_sell_loss = (signal['signal'] == -1) & (~is_sell_profit)

        # profit = pd.DataFrame(index = signal.index)
        signal['profit'] = 0

        # Buy profit or loss
        signal.loc[is_buy_profit, 'profit'] = signal['tp'] - signal['close'] - (signal['tp'] * comission) - (
                signal['close'] * comission)
        signal.loc[is_buy_loss, 'profit'] = signal['sl'] - signal['close'] - (signal['sl'] * comission) - (
                signal['close'] * comission)

        # Sell profit or loss
        signal.loc[is_sell_profit, 'profit'] = signal['close'] - signal['tp'] - (signal['close'] * comission) - (
                signal['tp'] * comission)
        signal.loc[is_sell_loss, 'profit'] = signal['close'] - signal['sl'] - (signal['close'] * comission) - (
                signal['sl'] * comission)

        return signal

    @staticmethod
    def with_drawdown(df):
        # todo: remove -10
        # profits = df.loc[df['profit'] != 0, 'profit'][:-10]
        profits = df['profit']
        max_drawdown = 0
        cur_drawdown = 0
        drawdowns = []

        for profit in profits.values:
            if profit < 0:
                # Loss - increase drawdown
                cur_drawdown -= profit  # increase drawdown
            else:
                # We have profit! Decrease drowdown, or if profit covered previous drawdown, set drawdown to 0.
                cur_drawdown = max(0, cur_drawdown - profit)
            max_drawdown = max(max_drawdown, cur_drawdown)
            drawdowns.append(cur_drawdown)

        df['drawdown'] = drawdowns
        return df

    @staticmethod
    def create_model_lgb(lgb_params):
        """ MultiOutputRegressor with lgb model. """

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        model = MultiOutputRegressor(lgb_model)
        print(f'Created new model {model}')
        return model

    @staticmethod
    def fit_predict_model(model, x_pipe, y_pipe, x_train, y_train, x_test, y_test):
        model.fit(x_pipe.transform(x_train), y_pipe.transform(y_train))

        # Predict and inverse transform to dataframe
        y_pred = model.predict(x_pipe.transform(x_test))
        y_pred = y_pipe.inverse_transform(y_pred)
        y_pred = pd.DataFrame(y_pred, columns = y_test.columns, index = y_test.index)
        return y_pred