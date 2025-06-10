import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover


input1: 'SPY'
dataF=yf.download('SPY', period='5y')
if isinstance(dataF.columns, pd.MultiIndex):
        dataF.columns = dataF.columns.get_level_values(0)

print(GOOG.head())
print(dataF.head())

def closing(data):
    return pd.Series(data)

def EWM(data, n):
     return pd.Series(data).ewm(span=n, adjust=True).mean()
     
# def RSI(dataF, days):
#     delta = dataF.Close.diff(1)
#     delta.dropna(inplace=True)
#     positive=delta.copy()
#     negative=delta.copy()

#     positive[positive < 0] = 0
#     negative[negative > 0] = 0

#     days = 14

#     average_gain = positive.rolling(window=days).mean()
#     average_loss = abs(negative.rolling(window=days).mean())
#     relative_strength = average_gain/average_loss

#     RSI_var = 100 - (100 / (1 + relative_strength))

#     combined = pd.DataFrame()
#     combined['RSI'] = RSI_var

#     return pd.Series(RSI_var)

class golden_setup(Strategy):
    # RSI_span = 14
    short_EWM_span = 50
    long_EWM_span = 200


    def init(self):
        # self.RSI = self.I(RSI, self.data, self.RSI_span)
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)

    def next(self):
        #Actual strategy if x: buy, elif y: sell
        #RSI Strategy: buy (stock=lower lows + RSI=comparitive higher lows) / sell (stock higher highs + RSI=comparative lower highs)
        try:
            if crossover(self.short_EWM[-1], self.long_EWM[-1]) or self.short_EWM[-1] > self.long_EWM[-1] and np.diff(self.short_EWM, prepend=0)[-1] > np.diff(self.long_EWM, prepend=0)[-1] or crossover(self.RSI[-1], 30):
                self.position.close()
                self.buy()
            elif self.short_EWM[-1] < self.long_EWM[-1]:
                self.position.close()
                self.sell()
        except:
             return




bt = Backtest(dataF, golden_setup, cash=10_000, commission=0.0)
stats = bt.run()
print(stats)
bt.plot()
