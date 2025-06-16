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

dataF=yf.download('GC=F', period='5y')
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

def trendline_intersect(data, short_EWM, long_EWM, days, intersect_distance):
    x = np.array(range(0,days))
    trend_buys = [0]*days
    for i in range(days,len(data)):
        short_coef = np.polyfit(x, short_EWM[i-days:i],1)
        long_coef = np.polyfit(x, long_EWM[i-days:i],1)
        correlation = np.corrcoef(x, short_EWM[i-days:i])[0,1]
        if short_coef[0] != long_coef[0]:
            intersection = -1*(long_coef[1]-short_coef[1])/(long_coef[0]-short_coef[0])

        if correlation**2 > 0.95 and intersection <= days+intersect_distance and intersection > 1 and short_EWM[i] > long_EWM[i]:
            trend_buys.append(True)
        else:
            trend_buys.append(False)
    return trend_buys

def EWM_gap(data, short_EWM, long_EWM, up_percent, down_percent):
    up_gap = []
    down_gap = []
    for i in range(len(data)):
        short_slope = [1,np.diff(short_EWM,prepend=0)[i]]
        long_slope = [1,np.diff(long_EWM,prepend=0)[i]]
        slope_sim = 1- (np.dot(short_slope/np.linalg.norm(short_slope),long_slope/np.linalg.norm(long_slope))**2)
        up_gap.append(data[-i]*up_percent*slope_sim)
        down_gap.append(data[-i]*down_percent*slope_sim)
    return [up_gap,down_gap]

    

##SELL IF WE ARE A FEW STANDARD DEV ABOVE TYPICAL RETURN SELL [A LOT MORE RESEARCH BECAUSE WE COULD JUST BE MISSING OUT ON MONEY]



class golden_setup(Strategy):
    RSI_buy_weight = 0#155
    RSI_sell_weight = 51
    EWM_buy_weight = 187
    EWM_sell_weight = 163
    trend_weight = 20

    #TRENDLINE
    intersect_dist = 40
    trend_length = 15

    #RSI
    RSI_span = 13
    RSI_upper = 80
    RSI_lower = 20

    #EWM
    short_EWM_span = 60
    long_EWM_span = 175
    EWM_up_percent = 0.1
    EWM_down_percent = 0.4

    n = 7


    def init(self):
        self.RSI = self.I(RSI, self.data.Close, self.RSI_span)
        self.short_EWM = self.I(EWM, self.data.Close, self.short_EWM_span)
        self.long_EWM = self.I(EWM, self.data.Close, self.long_EWM_span)
        self.trend = self.I(trendline_intersect, self.data.Close, self.short_EWM, self.long_EWM, self.trend_length, self.intersect_dist)
        self.EWM_gap = self.I(EWM_gap, self.data.Close,self.short_EWM, self.long_EWM, self.EWM_up_percent, self.EWM_down_percent)

    def next(self):
        EWM_buy =  self.short_EWM[-1] > self.long_EWM[-1]+self.EWM_gap[0][-1] and np.diff(self.short_EWM, prepend=0)[-1] > np.diff(self.long_EWM, prepend=0)[-1]
        EWM_below = self.short_EWM[-1] + self.EWM_gap[1][-1] < self.long_EWM[-1]
        RSI_buy = self.RSI[-1] < self.RSI_lower and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) < 0 and np.diff(self.RSI, prepend=0)[-1] > 0
        RSI_sell = self.RSI[-1] > self.RSI_upper and np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]) > 0 and np.diff(self.RSI, prepend=0)[-1] < 0
        #RSI_score = abs(np.dot([np.diff(self.RSI, prepend=0)[-1], 1]/np.linalg.norm([np.diff(self.RSI, prepend=0)[-1], 1]),[np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1]/np.linalg.norm([np.mean(np.diff(self.data.Close,prepend=0)[-self.n:-1]),1])))

        condition = self.EWM_buy_weight*(EWM_buy+(crossover(self.short_EWM, self.long_EWM))) - self.EWM_sell_weight*crossover(self.long_EWM,self.short_EWM) + self.RSI_buy_weight*RSI_buy - self.RSI_sell_weight*RSI_sell - self.trend_weight*(self.trend[-1])+self.EWM_buy_weight*EWM_below
        #RSI IS NOW PROPERLY IMPLEMENTED BUT BECAUSE WE STILL BUY EVERYTHING WAY TO QUICKLY IT'S EFFECTS CANNOT BE SEEN
        if condition>=100:
                self.buy(size=0.33)
                
        elif condition<-1:
                self.position.close()

bt = Backtest(dataF, golden_setup, cash=10_000, commission=0.0,exclusive_orders=False,finalize_trades=False)
stats = bt.run()
print(stats)
bt.plot()

#stats, heatmap, optimize_result = bt.optimize(
#    RSI_buy_weight = [0,200],
#    RSI_sell_weight = [0,200],
#    EWM_buy_weight = [0,200],
#    EWM_sell_weight = [0,200],
#    n = [0,10],
#    maximize='Equity Final [$]',
#    method='sambo',
#    max_tries=40,
#    random_state=0,
#    return_heatmap=True,
#    return_optimization=True)
#print(heatmap.sort_values().iloc[-3:])
