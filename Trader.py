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
    RSI_upper = 70
    RSI_lower = 30
    short_EWM_span = 50
    long_EWM_span = 200
    n = 5


    def init(self):
        self.RSI = self.I(RSI, self.data.Close, self.RSI_span)
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)

    def next(self):
        EWM_condition =  self.short_EWM[-1] > self.long_EWM[-1] and np.diff(self.short_EWM, prepend=0)[-1] > np.diff(self.long_EWM, prepend=0)[-1]
        RSI_buy = self.RSI[-1] < self.RSI_lower and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) < 0 and np.diff(self.RSI, prepend=0)[-1] > 0
        RSI_sell = self.RSI[-1] > self.RSI_upper and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) > 0 and np.diff(self.RSI, prepend=0)[-1] < 0
        RSI_weight = np.dot([np.diff(self.RSI, prepend=0)[-1], 1]/np.linalg.norm([np.diff(self.RSI, prepend=0)[-1], 1]),[np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1]/np.linalg.norm([np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1]))

        condition = 0.5*EWM_condition +0.5*(crossover(self.short_EWM[-1], self.long_EWM[-1]))+ RSI_weight*RSI_buy - 1*RSI_weight*RSI_sell - 1*(self.short_EWM[-1] < self.long_EWM[-1])
        print(condition)
        #RSI IS NOW PROPERLY IMPLEMENTED BUT BECAUSE WE STILL BUY EVERYTHING WAY TO QUICKLY IT'S EFFECTS CANNOT BE SEEN
        if condition>=0.5:
                self.buy(size=1)
                
        elif condition<0:
                self.position.close()

bt = Backtest(dataF, golden_setup, cash=10_000, commission=0.0,exclusive_orders=False,finalize_trades=True)
stats = bt.run()
print(stats)
bt.plot()
