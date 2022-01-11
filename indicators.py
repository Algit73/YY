#================================================================
#
#   File name   : indicators.py
#   Author      : PyLessons
#   Created date: 2021-01-20
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Used to plot 5 indicators with OHCL bars
#
#================================================================
import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator, ichimoku_a,ichimoku_b,ichimoku_base_line,ichimoku_conversion_line
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import rsi, williams_r
#from utils import Plot_OHCL

def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    '''df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()'''

    # Add Commodity Channel Index (CCI) indicator
    #df['CCI'] = cci(high=df["High"], low=df["Low"], close=df["Close"])

    # Add Average True Range (ATR) indicator
    #indicator_atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], fillna=True)
    #df['ATR'] = indicator_atr.average_true_range()


    # Add Moving Average Convergence Divergence (MACD) indicator
    #df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas

    # Add Relative Strength Index (RSI) indicator
    #df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas

    # Add Williams R% indicator
    df["Williams_R"] = williams_r(high=df["High"],low=df["Low"],close=df["Close"],lbp=14)

    #Add Ichimoku indicator
    #indicator_ichi = IchimokuIndicator(high=df["High"], low=df["Low"])
    '''df['ichi_a'] = ichimoku_a(high=df["High"], low=df["Low"])
    df['ichi_b'] = ichimoku_b(high=df["High"], low=df["Low"])
    df['ichi_base_line'] = ichimoku_base_line(high=df["High"], low=df["Low"])
    df['ichi_conversion_line'] = ichimoku_conversion_line(high=df["High"], low=df["Low"])'''
    
    return df

if __name__ == "__main__":   
    df = pd.read_csv('./Binance_BTCUSDT_1h.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)

    test_df = df[-400:]

    #Plot_OHCL(df)
