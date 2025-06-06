import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG

dataF=yf.download('SPY', period='5y')
if isinstance(dataF.columns, pd.MultiIndex):
        dataF.columns = dataF.columns.get_level_values(0)

print(GOOG.head())
print(dataF.head())

def closing(data):
    return pd.Series(data)

class BB(Strategy):

    def init(self):
        self.close = self.I(closing, self.data.Close)

    def next(self):
        #Actual strategy if x: buy, elif y: sell
        if self.close[-1] < self.close[-2]:
            self.position.close()
            self.buy()
        if self.close[-1] > self.close[-2]:
            self.position.close()
            self.sell()




bt = Backtest(dataF, BB, cash=10_000, commission=0.0)
stats = bt.run()
print(stats)
bt.plot()
