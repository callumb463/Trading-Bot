import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.express as px
import cufflinks as cf
import plotly.graph_objects as go

from typing import List
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import defaultdict
from collections.abc import Callable


def log_return(df):
    log_df = np.log(df['Close']/df['Close'].shift(1))
    log_df.columns = pd.MultiIndex.from_product([['Log Return'], log_df.columns])
    return log_df



class Trade:
    def __init__(self, buy_price, buy_index, buy_date):
        self.buy_price = buy_price
        self.buy_index = buy_index
        self.buy_date = buy_date
        self.sell_date = None
        self.price = buy_price

        ## Might be worth adding a list for multiple entry points 


        self.cum_log_return = 0
        self.duration = 0

        self.is_closed = False

    def update(self, log_return = None, current_price = None):
        if log_return is not None:
            self.cum_log_return += log_return
            self.price = self.price*np.exp(log_return)
        elif current_price is not None:
            self.cum_log_return += np.log(current_price/self.price)
            self.price = current_price
        self.duration += 1
    
    def close(self, sell_index, sell_date):
        self.is_closed = True
        self.sell_date = sell_date





class Strategy:
    def __init__(self, data, cash:int):
        self.cash = cash
        self.starting_cash = cash
        self.date = None
        self.data = data

        self.indicators = pd.DataFrame(index=self.data.index)
        self.indicator(log_return, self.data, names=['Log Return'])

        self.all_trades = []
        self.open_trades = {}
    
    def indicator(self, func: Callable, *args, names: List):
        func_data = func(*args)
        for name in names:
            self.indicators[name] = func_data[names]
        ## NEEDS TO HANDLE 2D DATA. MAYBE DITCH NUMPY IN FAVOR OF PANDAS


    def buy(self, buy_index, buy_price = None, portfolio_perc = None):
        if portfolio_perc is not None:
            buy_price = self.cash*portfolio_perc
        if int(buy_price) < int(self.cash):
            if buy_index in self.open_trades:
                current_trade = self.open_trades[buy_index]
                current_trade_equity = current_trade.price
                current_trade.close(buy_index, self.date)
                new_trade = Trade(buy_price=(buy_price+current_trade_equity), buy_index=buy_index, buy_date=self.date)
                self.all_trades.append(new_trade)
                self.open_trades.update({buy_index: new_trade})
                self.cash -= buy_price
            else:
                new_trade = Trade(buy_price=buy_price, buy_index=buy_index, buy_date=self.date)
                self.all_trades.append(new_trade)
                self.open_trades.update({buy_index: new_trade})
                self.cash -= buy_price
    
    def sell(self, sell_index):
        if sell_index in self.open_trades:
            sold_trade = self.open_trades.pop(sell_index)
            self.cash += sold_trade.price
            sold_trade.close(sell_index,self.date)

    def __str__(self):
        return (f'{self.indicators.shape[1]} indicator strategy with ${self.starting_cash}')

    def run(self, index, row):
        self.date = index
        for ticker, trade in self.open_trades.items():
            trade.update(log_return=row['Log Return'])
        pass

# BUY AND HOLD COMPARISON IS DIFFICULT FOR MULTIPLE STOCKS
# DOES IT EVEN HAVE VALUE?




class Backtest:
    def __init__(self, dataframe: pd.DataFrame, strategy: Strategy):
        self.df = dataframe
        self.strategy = strategy
        self.spy_return = 0
        self.equity = []

    def run_backtest(self):
        for index, row in self.strategy.indicators.iterrows():
            self.strategy.run(index, row)
            ##COUNT ACTUAL EQUITY NOT JUST CASH
            asset_price = 0
            for index, trade in self.strategy.open_trades.items():
                asset_price += trade.price
        
            self.equity.append(self.strategy.cash + asset_price)

    def visualize(self):

        # Price Plot
        fig = go.Figure()
        for ticker in self.df['Close']:
            fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'][ticker],name=ticker))
        for trade in self.strategy.all_trades:
            if isinstance(trade, Trade):
                fig.add_trace(go.Scatter(x=[trade.buy_date], y=[self.df['Close'][ticker][trade.buy_date]], mode='markers', marker=dict(size=10, color='green'),showlegend=False))
                fig.add_trace(go.Scatter(x=[trade.sell_date], y=[self.df['Close'][ticker][trade.sell_date]], mode='markers', marker=dict(size=10, color='red'),showlegend=False))
        fig.update_layout(title='Assets', xaxis_title='Date', yaxis_title='Price')
        fig.show()

        # Equity Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y = self.equity))
        fig.update_layout(title='Equity', xaxis_title='Date', yaxis_title='Cash')
        fig.show()

        # Trade Plot
        fig = go.Figure()
        for trade in self.strategy.all_trades:
            if isinstance(trade, Trade):
                if trade.price/trade.buy_price > 1:
                    fig.add_trace(go.Scatter(x=[trade.sell_date], y=[((trade.price/trade.buy_price)-1)*100], mode='markers', marker=dict(size=10, color='green'),showlegend=False))
                else:
                    fig.add_trace(go.Scatter(x=[trade.sell_date], y=[((trade.price/trade.buy_price)-1)*100], mode='markers', marker=dict(size=10, color='red'),showlegend=False))
        fig.update_layout(title='Trades', xaxis=dict(title='Sell Date'), yaxis=dict(title='Return', ticksuffix='%', zeroline=True, zerolinecolor='black', zerolinewidth=2))
        fig.show()
    # NEED TO WORK THIS OUT

    def __str__(self):
        return (f'Backtest of {self.strategy} with {len(self.strategy.all_trades)} trades')





class Stat:
    def __init__(self):
        self.stats = pd.Series({key: None for key in ["Start", "End", "Duration", 
            "Initial Equity", "Final Equity", "Peak Equity",
            "Return", "Volatility", "Sharpe Ratio", "Max Drawdown", 
            "Number of Trades", "Win Rate", "Avg PNL"]})
    def __add__(self, other):
        if isinstance(other, Stat):
            new_stats = pd.Series()
            if self.stats['Start'] == other.stats['Start']:
                new_stats['Start'] = self.stats['Start']
            else:
                new_stats['Start'] = 'Undefined'
            if self.stats['End'] == other.stats['End']:
                new_stats['End'] = self.stats['End']
            else:
                new_stats['End'] = 'Undefined'
            if self.stats['Duration'] == other.stats['Duration']:
                new_stats['Duration'] = self.stats['Duration']
            else:
                new_stats['Duration'] = 'Undefined'

            new_stats['Initial Equity'] = self.stats['Initial Equity'] + other.stats['Initial Equity']
            new_stats['Final Equity'] = self.stats['Final Equity'] + other.stats['Final Equity']
            new_stats['Max_Equity'] = 'Unfinished'
            ##FINISH MAX EQUITY AND EVERYTHING ELSE
            new_stats['Return'] = new_stats['Final Equity']/new_stats['Initial Equity']-1

            return new_stats
        else:
            raise TypeError("Unsupported operand type for +")
        
    def __str__(self):
        return self.stats
        

#TO IMPLEMENT:
#Monte Carlo Sim
#GBM Stock Sim
#Summative Stats



