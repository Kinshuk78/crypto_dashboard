from typing import Any
import requests
from pandas import DataFrame, to_datetime, Series
from collections import defaultdict
import streamlit as st

def create_returns(
    df : DataFrame,
    col : str = "close"
) -> DataFrame:
    df_copy = df.copy()
    if col not in df_copy.columns:
        raise ValueError(f"col '{col}' is not in the DataFrame")
    price_changes_dict = {
        "daily_returns" : 1,
        "weekly_returns" : 7,
        "monthly_returns" : 30
    }
    price_change_cols = {
        key : df_copy['close'].pct_change(periods = value).bfill() for key, value in price_changes_dict.items()
    }
    df_enriched = df_copy.assign(**price_change_cols)
    return df_enriched

def create_tecnical_indicators(
    df : DataFrame
) -> DataFrame:
    df_copy = df.copy()
    if "close" not in df_copy.columns:
        raise KeyError(f"the column 'close' is not in the DataFrame")
    df_copy['sma_7'] = df_copy['close'].rolling(window = 7).mean().bfill()
    df_copy['ema_7'] = df_copy['close'].ewm(span = 7, adjust = False).mean().bfill()
    return df_copy
    
 
class CriptoInfo:
    def __init__(
        _self,
        token : str = "ethereum"
    ):
        _self.token = token
        _self.headers = {
            "x-cg-demo-api-key" : "CG-wMi2kd693Povdsi3MEU3pDGz"
        }
        # placeholders
        _self.raw_prices = None
        _self.prices = None
        _self.raw_additional = None
        _self.market_data = None
    @st.cache_data
    def fetch_ohlc(
        _self,
        periods : int = 30 # 7, 14, 30, 90, 180, 365 possible values
    ):
        url = f"https://api.coingecko.com/api/v3/coins/{_self.token}/ohlc"
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
                headers = _self.headers
            )
            response = response.json()
            _self.raw_prices = response
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
    def modify_raw_prices(
        _self
    ):
        if _self.raw_prices is None:
            raise ValueError(f"There is no price data")
        # generate the dataframe
        data = defaultdict(list)
        cols = ['timestamp', 'open', 'high', 'low', 'close']
        for row in _self.raw_prices:
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
        _self.prices = filtered_data
    @st.cache_data
    def fetch_additional_info(
        _self,
        periods : int = 30
    ):
        url = f"https://api.coingecko.com/api/v3/coins/{_self.token}/market_chart"
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
                headers = _self.headers
            )
            response = response.json()
            _self.raw_additional = response
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
    def generate_input(
        _self
    ) -> DataFrame:
        if _self.raw_additional is None:
            raise ValueError(f"There is no additional info")
        if _self.raw_prices is None:
            raise ValueError(f"There is no price data")
        volume = _self.raw_additional.get("total_volumes", [])
        volume_data = defaultdict[Any, list](list)
        marketcap = _self.raw_additional.get('market_caps', [])
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
        temp = _self.prices.merge(
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
        _self
    ):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        querystring = {
            "vs_currency" : "usd",
            "ids" : _self.token,
            "price_change_percentage" : "24h"
        }
        try:
            response = requests.get(
                url = url,
                params = querystring,
                headers = _self.headers
            )
            if response.status_code != 200:
                raise Exception(f"Error due to invalid query")
            data = response.json()[0]
            _self.market_data = {
                "current_price" : data.get("current_price", 0),
                "market_cap" : data.get("market_cap", 0),
                "volume" : data.get("total_volume", 0),
                "24h_price_change" : data.get("price_change_percentage_24h_in_currency", 0)
            }
        except Exception as e:
            raise Exception(f"Error during fetching: '{e}'")
