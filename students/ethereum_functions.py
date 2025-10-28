from typing import Any
import requests
from pandas import DataFrame, to_datetime, Series
from collections import defaultdict
import streamlit as st
import numpy as np

def create_daily_returns(
    df : DataFrame,
    col : str = "close"
) -> DataFrame:
    df_copy = df.copy()
    if col not in df_copy.columns:
        raise ValueError(f"col '{col}' is not in the DataFrame")
    df_copy['daily_returns'] = df_copy[col].pct_change(periods = 1).bfill()
    return df_copy

def create_tecnical_indicators(
    df : DataFrame
) -> DataFrame:
    df_copy = df.copy()
    if "close" not in df_copy.columns:
        raise KeyError(f"the column 'close' is not in the DataFrame")
    df_copy['sma_7'] = df_copy['close'].rolling(window = 7).mean().bfill()
    df_copy['ema_7'] = df_copy['close'].ewm(span = 7, adjust = False).mean().bfill()
    delta = df_copy['close'].diff().bfill()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, delta, 0)
    mean_gain = Series(gain).rolling(window = 14).mean()
    mean_loss = Series(loss).rolling(window = 14).mean()
    rs = mean_gain / mean_loss
    df_copy['rsi_14'] = 100 - (100 / (1 + rs))
    return df_copy
    
 
class CriptoInfo:
    def __init__(
        self,
        token : str = "ethereum"
    ):
        self.token = token
        self.headers = {
            "x-cg-demo-api-key" : "CG-wMi2kd693Povdsi3MEU3pDGz"
        }
        # placeholders
        self.raw_prices = None
        self.prices = None
        self.raw_additional = None
        self.market_data = None
    @st.cache_data
    def fetch_ohlc(
        self,
        periods : int = 30 # 7, 14, 30, 90, 180, 365 possible values
    ):
        url = f"https://api.coingecko.com/api/v3/coins/{self.token}/ohlc"
        querystring = {
            "vs_currency" : "usd",
            "days" : str(periods),
            "precision" : "full"
        }
        try:
            # fetch the data from coingecko api
            response = requests.get(
                url = url,
                params = querystring,
                headers = self.headers
            )
            response = response.json()
            self.raw_prices = response
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
    def modify_raw_prices(
        self
    ):
        if self.raw_prices is None:
            raise ValueError(f"There is no price data")
        # generate the dataframe
        data = defaultdict(list)
        cols = ['timestamp', 'open', 'high', 'low', 'close']
        for row in self.raw_prices:
            date, open, high, low, close = row
            data[cols[0]].append(date)
            data[cols[1]].append(open)
            data[cols[2]].append(high)
            data[cols[3]].append(low)
            data[cols[4]].append(close)
        data = DataFrame(
            data = data
        )
        # create dates column
        data['timestamp'] = to_datetime(data['timestamp'], unit = "ms")
        # filter the dates
        filtered_data = data.loc[data.groupby(data['timestamp'].dt.date)['timestamp'].idxmax()]
        filtered_data.reset_index(drop = True, inplace = True)
        filtered_data['timestamp'] = filtered_data['timestamp'].dt.date
        self.prices = filtered_data
    @st.cache_data
    def fetch_additional_info(
        self,
        periods : int = 30
    ):
        url = f"https://api.coingecko.com/api/v3/coins/{self.token}/market_chart"
        querystring = {
            "vs_currency" : "usd",
            "days" : str(periods),
            "interval" : "daily",
            "precision" : "full"
        }
        try:
            # fetch the data from coingecko api
            response = requests.get(
                url = url,
                params = querystring,
                headers = self.headers
            )
            response = response.json()
            self.raw_additional = response
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
    def generate_input(
        self
    ) -> DataFrame:
        if self.raw_additional is None:
            raise ValueError(f"There is no additional info")
        if self.raw_prices is None:
            raise ValueError(f"There is no price data")
        volume = self.raw_additional.get("total_volumes", [])
        volume_data = defaultdict[Any, list](list)
        marketcap = self.raw_additional.get('market_caps', [])
        marketcap_data = defaultdict[Any, list](list)
        if len(volume) > 0:
            for row in volume:
                volume_data['timestamp'].append(row[0])
                volume_data['volume'].append(row[1])
            volume_data = DataFrame(volume_data)
        else:
            raise ValueError(f"There is no volume data")
        if len(marketcap) > 0:
            for row in marketcap:
                marketcap_data['timestamp'].append(row[0])
                marketcap_data['marketcap'].append(row[1])
            marketcap_data = DataFrame(marketcap_data)
        else:
            raise ValueError(f"There is no marketcap data")
        # modify the dates
        volume_data['timestamp'] = to_datetime(volume_data['timestamp'], unit = "ms")
        volume_data['timestamp'] = volume_data['timestamp'].dt.date
        marketcap_data['timestamp'] = to_datetime(marketcap_data['timestamp'], unit = "ms")
        marketcap_data['timestamp'] = marketcap_data['timestamp'].dt.date
        # join all the information
        temp = self.prices.merge(
            right = volume_data,
            how = "inner",
            on = "timestamp"
        )
        temp = temp.merge(
            right = marketcap_data,
            how = "inner",
            on = "timestamp"
        )
        return temp
    @st.cache_data
    def get_market_data(
        self
    ):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        querystring = {
            "vs_currency" : "usd",
            "ids" : self.token,
            "price_change_percentage" : "24h"
        }
        try:
            response = requests.get(
                url = url,
                params = querystring,
                headers = self.headers
            )
            if response.status_code != 200:
                raise Exception(f"Error due to invalid query")
            data = response.json()[0]
            self.market_data = {
                "current_price" : data.get("current_price", 0),
                "market_cap" : data.get("market_cap", 0),
                "volume" : data.get("total_volume", 0),
                "24h_price_change" : data.get("price_change_percentage_24h_in_currency", 0)
            }
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
