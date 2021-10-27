import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import datetime
from indi import indicator

pd.options.mode.chained_assignment = None


symbols = [' ','BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'DOTUSDT', 'XRPUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT',
           'BCHUSDT', 'XLMUSDT', 'LUNAUSDT', 'DOGEUSDT', 'VETUSDT','ATOMUSDT', 'AAVEUSDT', 'FILUSDT', 'AVAXUSDT',
           'TRXUSDT', 'EOSUSDT', 'SOLUSDT', 'IOTAUSDT', 'XTZUSDT', 'NEOUSDT', 'CHZUSDT', 'DAIUSDT', 'SNXUSDT',
           'SUSHIUSDT', 'EGLDUSDT', 'ENJUSDT', 'ZILUSDT', 'MATICUSDT', 'MKRUSDT', 'COMPUSDT', 'BATUSDT', 'ZRXUSDT',
           'RSRUSDT'
           ]


timef = [ '12h', '1d','30m', '1h','4h']

## CREATE indicators
def concatenate_2_lists_diff_sizes(list1, list2):
    return [x + str(y) for x in list1 for y in list2]

indi_ma = ['SMA', 'EMA', 'WMA', 'BBANDS']
range_indi_ma = [20,30,40,50,100,150,200]

#create lists of all moving average indicators
indicators_ma = concatenate_2_lists_diff_sizes(indi_ma, range_indi_ma)
indicators_ma.append('ICHIMOKU')
others_indi =[ 'StochRSI','RSI' , 'MACD','OBV', 'STOCHASTIC_OSCILLATOR']
indic_others = {k: [i for i in others_indi if i.startswith(k)] for k in others_indi}

    


##RANDOM COLORS FOR EACH  MA INDICATORS
colors_sma = ['#1f77b4',  # muted blue
              '#ff7f0e',  # safety orange
              '#2ca02c',  # cooked asparagus green
              ]
colors_wma = ['#d62728',  # brick red
              '#9467bd',  # muted purple
              '#8c564b',  # chestnut brown
              ]
colors_ema = ['#e377c2',  # raspberry yogurt pink
              '#7f7f7f',  # middle gray
              '#bcbd22',  # curry yellow-green
              ]



## STREAMLIT INPUT AND SIDEBAR
st.title('Play with some indicators')
st.text('This app lets you analyze price movements of major cryptocurrencies. On the side bar you can choose a starting date,')
st.text('an ending date and you have a choice between multiple timeframes and indicators. Let')
st.text('me know if you would like to see other indicators, timeframes, crypto or features !')
symbol = st.sidebar.selectbox('Search Pairs', symbols)
start_date = st.sidebar.date_input('Start date', datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.date.today())
timeframes = st.sidebar.selectbox('Timeframe', timef)
indicators_ma = st.sidebar.multiselect('Indicators on price chart', indicators_ma)
indicators_others = st.sidebar.multiselect('Other Indicators', others_indi)

#Display charts with indicators from indi.py

def display(symbol, timeframes, indicators_ma, indicators_others, start_date, end_date):
    
    
    text = st.write(f""" # {symbol} Price *chart*""")
    df = pd.read_csv(f'data/{timeframes}/{symbol}-{timeframes}-data.csv', parse_dates=['timestamp'], index_col='timestamp')
    

    if start_date < end_date:
        df = df[start_date : end_date]
    else:
        st.error('Error: End date must fall after start date.')



    n_rows = len([i for i in indic_others.values() if len(i) > 0])

    fig = make_subplots(rows = 2 + n_rows, cols = 1, shared_xaxes = True, vertical_spacing = 0.02, row_heights = [1,0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    row_idx = 1


    trace_price = fig.add_trace(
                        go.Candlestick(
                              x = df.index,
                              open = df['open'],
                              high = df['high'],
                              low = df['low'],
                              close = df['close'],
                              name = 'Candlestick'),
                        
                               
                              row = row_idx, col = 1
                    )
              
    colors = ['green' if row['open'] - row['close'] >= 0 
                       else 'red' for index, row in df.iterrows()]
    
    fig.add_trace(go.Bar(x=df.index, 
                         y=df['volume'],
                         name = 'Volume',
                         marker_color=colors), row=row_idx + 1, col=1)

                
    for indi in indicators_ma:
        if indi.startswith('SMA'):       
            sma_trace = go.Scatter( 
                          x=df.index, 
                          y=indicator.simple_moving_average(df,int(indi[3:])),
                          line=dict(color= np.random.choice(colors_sma, replace = False), width=1), 
                          name = f'SMA{indi[3:]}'
                      )
                    
            fig.add_trace(sma_trace, 
                          row = row_idx, col = 1)

        if indi.startswith('WMA'):            
            wma_trace = go.Scatter(
                          x=df.index,
                          y=indicator.weighted_moving_average(df,int(indi[3:])),
                          line=dict(color=np.random.choice(colors_wma, replace = False), width=1), 
                          name = f'WMA{indi[3:]}'
                      )

            fig.add_trace(wma_trace,
                         row = row_idx, col = 1) 

        if indi.startswith('EMA'):              
            ema_trace = go.Scatter(
                          x=df.index, 
                          y=indicator.exponential_moving_average(df,int(indi[3:])),
                          line=dict(color=np.random.choice(colors_ema, replace = False), width=1), 
                          name = f'EMA{indi[3:]}'
                      )

            fig.add_trace(ema_trace, 
                          row = row_idx, col = 1)
        
        if indi.startswith('BBANDS'):
            bands_up, bands_down = indicator.bollinger_bands(df, int(indi[6:]))
            
            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = bands_up,
                        line = dict(color = 'green', width = 1),
                        name = 'BBANDS_UP'), row = row_idx, col = 1
                  )
            
            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = bands_down,
                        line = dict( color = 'red', width = 1),
                        name = 'BBANDS_DOWN'), row = row_idx, col = 1
                  )
        if indi.startswith('ICHIMOKU'):
            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = indicator.ichimoku(df)

            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = tenkan_sen,
                        line = dict( width =  0.5),
                        name = 'tenkan_sen'), row = row_idx, col = 1
                  )
            
            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = kijun_sen,
                        line = dict(width =  0.5),
                        name = 'kijun_sen'), row = row_idx, col = 1
                  )

            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = senkou_span_a,
                        line = dict(width =  0.5),
                        fill= None,
                        name = 'senkou_span_a'), row = row_idx, col = 1
                  )
            
            fig.add_trace(
                  go.Scatter(
                        x = df.index, y = senkou_span_b,
                        line = dict( width =  0.5),
                        fill='tonexty',
                        name = 'senkou_span_b'), row = row_idx, col = 1
                  )
            




    for other in indicators_others:

        #OBV  
        if other.startswith('OBV'):
            fig.update_yaxes(title_text="OBV", row = row_idx+2, col = 1)
            obv_trace = go.Scatter(
                              x=df.index, y= indicator.on_balance_volume(df),
                              line=dict(color='red', width=1),
                              name = other
                        )
            fig.add_trace(obv_trace, row = row_idx+2, col = 1)


          
        #STOCH RSI                                  
        if other.startswith('StochRSI'):
            fig.update_yaxes(title_text="StochasticRSI", row = row_idx+2, col = 1)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=indicator.stoch_rsi(df),
                    name= other, marker_color= 'blue'
                    ), row= row_idx+2, col=1,
            )
                    
            fig.add_trace(
                go.Scatter(
                        x=df.index, y=[70] * len(df.index),
                        name='Overbought', marker_color='#109618',
                        line = dict(dash='dot'), showlegend=False,
                        ), row= row_idx+2, col=1,
            )
                    
            fig.add_trace(
                    go.Scatter(
                        x=df.index, y=[30] * len(df.index),
                        name='Oversold', marker_color='#109618',
                        line = dict(dash='dot'),showlegend=False,
                        ),row= row_idx+2, col=1,
            )
        #RSI 
        if other.startswith('RSI'):
            fig.update_yaxes(title_text="RSI", row = row_idx+2, col = 1)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=indicator.relative_strength_index(df),
                    name= other, marker_color= 'blue'
                    ), row= row_idx+2, col=1,
            )
                    
            fig.add_trace(
                go.Scatter(
                        x=df.index, y=[70] * len(df.index),
                        name='Overbought', marker_color='#109618',
                        line = dict(dash='dot'), showlegend=False,
                        ), row= row_idx+2, col=1,
            )
                    
            fig.add_trace(
                    go.Scatter(
                        x=df.index, y=[30] * len(df.index),
                        name='Oversold', marker_color='#109618',
                        line = dict(dash='dot'),showlegend=False,
                        ),row= row_idx+2, col=1,
            )
        
        

        #MACD
        if other.startswith('MACD'):
            fig.update_yaxes(title_text="MACD", row = row_idx+2, col = 1)

            macd, macd_s, macd_h = indicator.moving_average_convergence_divergence(df)
            #fast signal
            fig.add_trace(
                go.Scatter(
                        x = df.index, y = macd, 
                        line=dict(color='green', width=1), 
                        name = 'macd',
                        ), row = row_idx+2, col = 1,
            )
            #slow signal
            fig.add_trace(
                go.Scatter(
                        x = df.index, y = macd_s, 
                        line=dict(color='red', width=1), 
                        name = 'signal',
                        ), row = row_idx+2, col = 1,
            )
            # Colorize the histogram values
            colors = np.where(macd_h < 0, 'red', 'green')
            # Plot the histogram
            fig.add_trace(
                go.Bar(
                        x = df.index, y =  macd_h, 
                        name='histogram', marker_color=colors,
                        ), row = row_idx+2, col = 1, 
            )

        if other.startswith('STOCHASTIC_OSCILLATOR'):
            fig.update_yaxes(title_text="Stoch", row = row_idx+2, col = 1)
            K, d = indicator.stochastic_oscillator(df)

            # Fast Signal (%k)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=K,
                    line=dict(color='#ff9900', width=1),
                    name='fast',
                ), row= row_idx+2, col=1  
            )
            # Slow signal (%d)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=d,
                    line=dict(color='#000000', width=1),
                    name='slow'
                ), row= row_idx+2, col=1  
            )

                         

        # increase row index just if there are indicators in this group
        row_idx += 1


    fig.update_layout(width= 800,height = 1500, xaxis_rangeslider_visible=False, showlegend= True)
                      
       
    
    fig.update_xaxes(gridcolor ='#7f7f7f')
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price", row = 1, col = 1)
    fig.update_yaxes(title_text="Volume", row = 2, col = 1)
    
    fig.update_yaxes(gridcolor ='#7f7f7f')

    chart = st.plotly_chart(fig, width= 800,height = 1500 * n_rows)
    

    

    return chart

display(symbol, timeframes, indicators_ma, indicators_others, start_date, end_date)