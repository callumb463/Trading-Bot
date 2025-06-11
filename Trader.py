import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover

dataF=yf.download('AAPL', period='5y')
if isinstance(dataF.columns, pd.MultiIndex):
        dataF.columns = dataF.columns.get_level_values(0)


def closing(data):
    return pd.Series(data)

def EWM(data, n):
     return pd.Series(data).ewm(span=n, adjust=True).mean()

def RSI(data, days):
    delta = pd.Series(data).diff(1)
    positive=delta.copy()
    negative=delta.copy()

    positive[positive < 0] = 0
    negative[negative > 0] = 0

    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())
    relative_strength = average_gain/average_loss

    RSI_var = 100 - (100 / (1 + relative_strength))

    return pd.Series(RSI_var)
     
class golden_setup(Strategy):
    RSI_span = 14
    short_EWM_span = 50
    long_EWM_span = 200


    def init(self):
        self.RSI = self.I(RSI, self.data.Close, self.RSI_span)
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)

    def next(self):
        #EACH POSITION IS AN INSTANCE OF CLASS. JUST CREATE NEW ONES
        EWM_condition = crossover(self.short_EWM[-1], self.long_EWM[-1]) and np.diff(self.short_EWM, prepend=0)[-1] > np.diff(self.long_EWM, prepend=0)[-1]
        if  self.short_EWM[-1] < self.long_EWM[-1] or EWM_condition or crossover(self.RSI[-1], 30):
                self.buy(size=2)
                
        elif self.short_EWM[-1] > self.long_EWM[-1]:# or crossover(70, self.RSI[-1]):
                self.position.close()

bt = Backtest(dataF, golden_setup, cash=10_000, commission=0.0,exclusive_orders=False)
stats = bt.run()
print(stats)
bt.plot()
