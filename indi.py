import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np

class indicator:

        @st.cache
        def cumulative_moving_average(df, range_indi):
            df[f'CMA{range_indi}'] = df['close'].expanding(min_periods=range_indi).mean()
            return df[f'CMA{range_indi}']

        @st.cache
        def simple_moving_average(df, range_indi):
            df[f'SMA{range_indi}'] = df['close'].rolling(window = range_indi).mean()
            return df[f'SMA{range_indi}']

        @st.cache
        def exponential_moving_average(df, range_indi):
            df[f'EMA{range_indi}'] = df['close'].ewm(span = range_indi, adjust = False).mean()
            return df[f'EMA{range_indi}']

        @st.cache
        def weighted_moving_average(df, range_indi):
            weights = np.arange(1,range_indi+1)
            wma = df['close'].rolling(range_indi).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
            df[f'WMA{range_indi}'] = wma
            return df[f'WMA{range_indi}']

        @st.cache
        def bollinger_bands(df, range_indi):
            sma = df['close'].rolling(window = range_indi).mean()
            std = df['close'].rolling(window = range_indi).std()
            df['BANDS_UP'] = sma + 2 * std
            df['BANDS_DOWN'] = sma - 2 * std
            return df['BANDS_UP'], df['BANDS_DOWN']


        @st.cache   
        def moving_average_convergence_divergence(df):
            # Get the 26-day EMA of the closing price
            twenty_six_ema = df['close'].ewm(span = 12, adjust = False, min_periods = 12).mean()
            # Get the 12-day EMA of the closing price
            twelve_ema = df['close'].ewm(span = 26, adjust = False, min_periods = 26).mean()
            # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
            macd = twenty_six_ema - twelve_ema
            # Get the 9-Day EMA of the MACD for the Trigger line
            macd_s = macd.ewm(span = 9, adjust = False, min_periods = 9).mean()
            # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
            macd_h = macd - macd_s

            
            df['macd'] = macd 
            df['macd_s'] = macd_s 
            df['macd_h'] = macd_h

            return df['macd'],df['macd_s'],df['macd_h']

        
        @st.cache
        def relative_strength_index(df, periods = 14):
            df['rsi'] = ta.rsi(close = df['close'], length = periods)
            return df['rsi']

        
        @st.cache
        def on_balance_volume(df):
            '''
            Bullish divergence : when the price action decreases and OBV increases simultaneously, 
                                you can anticipate upward movements. Accordingly, for a bullish divergence, 
                                the price displays lower lows while the indicator shows higher lows.

            Bearish divergence :  identified when the price continues to rise while at the same time the OBV 
                                  indicator declines. When the price exhibits higher highs and on-balance volume 
                                  has lower highs, it’s a sign of a bearish divergence.

            Confirming the trend : to confirm the trend direction, see whether the OBV line moves in the same 
                                  direction as the price. If on-balance volume increases when price increases, 
                                  you can confirm an upward trend and the volume moves to support the price growth.

            Potential breakout or breakdown from the ranging market : during ranging market conditions, you should be on the lookout for a
                                                                     rising or decreasing on-balance volume indicator value, as it can a signal 
                                                                     a potential breakout or breakdown in price. An increasing OBV line can alert 
                                                                     you to a potential upward breakout because accumulation is in place.
            '''
            
            obv = []
            obv.append(0)
            #loop through dataset
            for i in range(1, len(df['close'])):
                if df['close'][i] > df['close'][i-1]:
                  obv.append(obv[-1] + df['volume'][i])
                elif df['close'][i] < df['close'][i-1]:
                  obv.append(obv[-1] - df['volume'][i])
                else:
                  obv.append(obv[-1])
            #store obv and ema in new columns
            df['obv'] = obv 
            
            return df['obv'] 

        


        @st.cache
        def stoch_rsi(df, length=14):
            df['stochrsi'] = ta.stochrsi(close = df['close'], length = length)
            return df['stochrsi']

        

        def stochastic_oscillator(df):
            df['14-low'] = df['low'].rolling(14).min()
            df['14-high'] = df['high'].rolling(14).max()
            df['K'] = (df['close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
            df['D'] = df['K'].rolling(3).mean()
            return df['K'],  df['D']


        def ichimoku(df):
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
            nine_period_high = df['high'].rolling(window= 9).max()
            nine_period_low = df['low'].rolling(window= 9).min()
            df['tenkan_sen'] = (nine_period_high + nine_period_low) /2

            # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
            period26_high = df['high'].rolling(window=26).max()
            period26_low = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (period26_high + period26_low) / 2

            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
            period52_high = df['high'].rolling(window=52).max()
            period52_low = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

            # The most current closing price plotted 26 time periods behind (optional)
            df['chikou_span'] = df['close'].shift(-26)

            return df['tenkan_sen'],df['kijun_sen'],df['senkou_span_a'],df['senkou_span_b'],df['chikou_span']
                    