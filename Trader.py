import yfinance as yf
import pandas as pd
import numpy as np
import sambo
import matplotlib.pyplot as plt
from sambo.plot import plot_objective
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover

dataF=yf.download('SBUX', period='5y')
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
    RSI_buy_weight = 154
    RSI_sell_weight = 20
    EWM_buy_weight = 186
    EWM_sell_weight = 162



    RSI_span = 13
    RSI_upper = 80
    RSI_lower = 20
    short_EWM_span = 60
    long_EWM_span = 175
    n = 7


    def init(self):
        self.RSI = self.I(RSI, self.data.Close, self.RSI_span)
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)

    def next(self):
        EWM_condition =  self.short_EWM[-1] > self.long_EWM[-1] and np.diff(self.short_EWM, prepend=0)[-1] > np.diff(self.long_EWM, prepend=0)[-1]
        RSI_buy = self.RSI[-1] < self.RSI_lower and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) < 0 and np.diff(self.RSI, prepend=0)[-1] > 0
        RSI_sell = self.RSI[-1] > self.RSI_upper and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) > 0 and np.diff(self.RSI, prepend=0)[-1] < 0
        RSI_score = np.dot([np.diff(self.RSI, prepend=0)[-1], 1]/np.linalg.norm([np.diff(self.RSI, prepend=0)[-1], 1]),[np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1]/np.linalg.norm([np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1]))

        condition = self.EWM_buy_weight*(EWM_condition+(crossover(self.short_EWM[-1], self.long_EWM[-1]))) - self.EWM_sell_weight*(self.short_EWM[-1] < self.long_EWM[-1]) + self.RSI_buy_weight*RSI_score*RSI_buy - self.RSI_sell_weight*RSI_score*RSI_sell
        #RSI IS NOW PROPERLY IMPLEMENTED BUT BECAUSE WE STILL BUY EVERYTHING WAY TO QUICKLY IT'S EFFECTS CANNOT BE SEEN
        if condition>=100:
                self.buy(size=1)
                
        elif condition<0:
                self.position.close()

bt = Backtest(dataF, golden_setup, cash=10_000, commission=0.0,exclusive_orders=False,finalize_trades=True)
stats = bt.run()
print(stats)
bt.plot()

#stats, heatmap, optimize_result = bt.optimize(
#    RSI_upper = [50,100],
#    RSI_lower = [0,50],
#    short_EWM_span = [0,100],
#    long_EWM_span = [100,300],
#    n = [0,10],
#    maximize='Equity Final [$]',
#    method='sambo',
#    max_tries=40,
#    random_state=0,
#    return_heatmap=True,
#    return_optimization=True)
#print(heatmap.sort_values().iloc[-3:])
