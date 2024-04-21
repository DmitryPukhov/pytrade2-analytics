import os
import glob
import pandas as pd
from datetime import timedelta,date, datetime

class DataTool:
    
    @staticmethod
    def read_last_candles(ticker, data_dir, days=1):
        """ Read 1min candles from data_dir for given days """
    
        file_paths = sorted([f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith(f"{ticker}_candles_1min.csv")])[-days:]
        data = pd.concat([pd.read_csv(f, parse_dates=["open_time", "close_time"]) for f in file_paths])
        data.set_index("close_time", drop=False, inplace=True)
        return data