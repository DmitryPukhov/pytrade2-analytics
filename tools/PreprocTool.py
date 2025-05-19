import pathlib

from features.level2.Level2Features import Level2Features
import numpy as np
import pandas as pd
import os

class PreprocTool:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.data_dir_raw = f"{self.data_dir}/raw"
        self.data_dir_preproc = f"{self.data_dir}/preproc"

    @staticmethod
    def level2_transform(df_level2_raw):
        # from features.level2.Level2Buckets import Level2Buckets
        # df_level2_buckets = Level2Buckets().level2_buckets(df_level2_raw, "60s")
        # df_level2_buckets.tail()
        df_level2 = Level2Features().expectation(df_level2_raw)
        #df_level2_features = df_level2_features.rolling("1min").agg("mean")
        df_level2["l2_bid_ask_vol"] = df_level2["l2_bid_vol"] + df_level2["l2_ask_vol"]
        #df_level2 = df_level2[["l2_bid_expect", "l2_bid_vol","l2_ask_expect", "l2_ask_vol", "l2_bid_ask_vol"]]
        return df_level2
    
    def get_unprocessed_raw_files( self, kind):
        """ 
        Compare raw files dir with preprocessed dir, get not processed raw files 
        Raw files have .csv.zip extension, processed are with .csv
        """
        
        raw_files = sorted(os.listdir(os.path.join(self.data_dir_raw, kind)))
        raw_files = [f for f in raw_files if f.endswith(f"{kind}.csv.zip")]
        if not raw_files: 
            return []
    
        # Get all raw and preprocessed files
        raw_files_wo_zip = sorted([pathlib.Path(raw_file).stem for raw_file in raw_files])
        preproc_files = sorted(os.listdir(os.path.join(self.data_dir_preproc, kind)))
        preproc_files = [f for f in preproc_files if f.endswith(f"{kind}.csv")]
        if not preproc_files:
            return raw_files
    
        # Compare raw with preprocessed, create unprocessed list, redo last preprocessed (it could be not full previously)
        last_processed_raw_file = f"{preproc_files[-1]}.zip"
        last_raw_file = f"{raw_files[-1]}"
        if last_raw_file == last_processed_raw_file:
            print(f"No new raw data appeared after {last_raw_file}")
            return []
        
        files_unprocessed_wo_zip = set(raw_files_wo_zip) - set(preproc_files)
        raw_files_unprocessed = sorted([f"{f}.zip" for f in files_unprocessed_wo_zip] + [last_processed_raw_file])
    
        return raw_files_unprocessed
    
    def preprocess_last_raw_data(self, ticker:str, kind:str, days:int = 1, datetime_col: str = "datetime", agg = "mean", transform_func = None):
        """ Read raw data resampled to 1 min. Needed to reduce data amount"""
        source_dir = f"{self.data_dir_raw}/{kind}"
        target_dir = f"{self.data_dir_preproc}/{kind}"
        # file_paths = sorted(
        #     [f"{source_dir}/{f}" for f in os.listdir(source_dir)])[-days:]
        
        unprocessed_raw_files = self.get_unprocessed_raw_files(kind)
        file_paths =  [f"{source_dir}/{f}" for f in unprocessed_raw_files]
        print(f"Preprocess {len(file_paths)} new {kind} raw files")
        for raw_file_path in file_paths:
            print(f"Read {ticker} {kind} data from {raw_file_path}")
            # Read raw data
            df = pd.read_csv(raw_file_path, parse_dates = True)
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, drop = False, inplace = True)
    
            # Transform to features if needed
            if transform_func:
                df = transform_func(df)
    
            #Drop duplicate datetime column
            drop_cols = ["datetime", f"{datetime_col}.1", "ticker", "symbol"]
            for drop_col in drop_cols:
                if drop_col in df.columns:
                    del df[drop_col]        
    
            df = df.resample("1min", label = "right", closed = "right").agg(agg)
            
            # Prepare target path
            target_file_name = pathlib.Path(raw_file_path).stem
            if not target_file_name.endswith("csv"): target_file_name += ".csv"
            preprocessed_file_path = os.path.join(target_dir, target_file_name)
    
            # Write
            print(f"Write {ticker} {kind} data to {preprocessed_file_path}")
            df.to_csv(preprocessed_file_path, header = True)
            
    def read_last_preproc_data(self, ticker:str, kind:str, days=1, datetime_col = "datetime"):
        """ Read last given days from preprocessed directory """
        
        source_dir = f"{self.data_dir_preproc}/{kind}"
        print(f"Read {ticker} {kind} data for {days} days from {source_dir}")
    
        file_paths = sorted(
            [f"{source_dir}/{f}" for f in os.listdir(source_dir) if f.endswith(".csv")])[-days:]
        
        df = pd.concat([pd.read_csv(f, parse_dates=True) for f in file_paths])
        df = self.clean_columns(df, datetime_col)
        return df
        
    def clean_columns(self, df: pd.DataFrame, datetime_col:str="datetime"):
        """ After level2, bidask or candles df has been read, set datetime index, clean columns which are not needed"""
        remove_cols = [f"{datetime_col}.1","ticker", "symbol"]
        for remove_col in remove_cols:
            if remove_col in df.columns:
                del df[remove_col]
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        if "close_time" in df.columns and "vol" in df.columns:
            # remove duplicated
            df = df.groupby(df["close_time"]) \
                .apply(lambda x: x[x['vol'] == x['vol'].max()]) \
                .reset_index(drop=True)
            #df = df.loc[df.groupby(df['close_time'])['vol'].idxmax()] #.reset_index(drop=True)
        df = df.set_index(datetime_col, drop = False, inplace = False)
        return df 
