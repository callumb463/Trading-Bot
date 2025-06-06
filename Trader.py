import yfinance as yf
import pandas as pd
from backtesting import Strategy
from backtesting import Backtest

dataF=yf.download('SPY', period='5y')
print(dataF)

class XXX(Strategy):

    def init(self):
        #Any functions (indicators) need to be initialised with self.#indicator# = self.I(#function#, #data#)
        return

    def next(self):
        #Actual strategy if x: buy, elif y: sell
        return




#bt = Backtest( #Stock , #Strategy , cash=10_000, commission=.002)
#stats = bt.run()
#stats
