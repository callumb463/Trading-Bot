import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover

dataF=yf.download('SPY', period='5y')
if isinstance(dataF.columns, pd.MultiIndex):
        dataF.columns = dataF.columns.get_level_values(0)

print(GOOG.head())
print(dataF.head())

def closing(data):
    return pd.Series(data)

def EWM(data, n):
     return pd.Series(data).ewm(span=n, adjust=True).mean()
     

class golden_setup(Strategy):
    short_EWM_span = 50
    long_EWM_span = 200


    def init(self):
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)

    def next(self):
        #Actual strategy if x: buy, elif y: sell
        try:
            if self.short_EWM[-1] > self.long_EWM[-1] or crossover(self.short_EWM[-1], self.longEWM[-1]):
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
