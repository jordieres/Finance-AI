'''@package DataPreprocessing
This module contains the DataProcessor class that processes the data and creates the output files. It also contains the DataManipulation class that contains methods to manipulate the data.
'''


import os, time, gc, sys, io
import datetime, pickle
import warnings, random, pdb

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as AA
import os
import pickle
import argparse
import yaml

from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Stock:
    def __init__(self, ticker, file_name, lahead, tr_tst):
        self.ticker = ticker
        self._data = self._read_data(file_name)
        self._lahead = lahead
        self._tr_tst = tr_tst
        self.df = pd.DataFrame(self._data)
        self.serial_dict = {}
        self.mserial_dict = {}

    def _read_data(self, file_name):
        '''
        Returns the data from the data path
        
        Parameters:
        file_name - path to the data
        '''
        return pd.read_csv(file_name, sep=",", index_col=0, parse_dates=True)

    def process_univariate_data(self, win):
        '''
        Parameters:
        - win (int): Window size.
        - tr_tst (float): Train-test ratio.
        '''
        for ahead in self._lahead:
            df = self.df

            win_x = sliding_window_view(df['PX_OPEN'].to_numpy(), window_shape=win)[::1]
            Y = df.iloc[ahead + win:]['PX_OPEN']
            X = pd.DataFrame.from_records(win_x)

            X = X.iloc[:-(ahead + 1), :]
            X.set_index(df.index[(win - 1):-(ahead + 1)], drop=True, inplace=True)

            xm = self.calculate_mean(X, axis=1)
            cX = X.sub(xm, axis=0)

            mnx = round(min(self.calculate_min(cX)))
            mxx = round(min(self.calculate_max(cX)))
            vdd = pd.DataFrame({'mean': xm, 'min': mnx, 'max': mxx})
            vdd.set_index(X.index)

            cXn = cX.apply(lambda x: (x - mnx) / (mxx - mnx), axis=1)
            cYn = pd.Series([((i - j) - mnx) / (mxx - mnx) for i, j in zip(Y.tolist(), xm.tolist())], index=Y.index)
            cXn = cXn.astype('float32')
            cYn = cYn.astype('float32')

            pmod = int(cXn.shape[0] * self._tr_tst)
            trainX = cXn.iloc[:pmod, :]
            trainY = cYn.iloc[:pmod]
            testX = cXn.iloc[pmod:, :]
            testY = cYn.iloc[pmod:]

            self.serial_dict[ahead] = {"x": X, "y": Y, "nx": cXn, "ny": cYn, "numt": pmod,
                                        "trainX": trainX, "trainY": trainY,
                                        "testX": testX, "testY": testY, "vdd": vdd}

    def process_multivariate_data(self, mwin, m_ftrs):
        '''
        Parameters:
        mwin - multivariate window size
        m_ftrs - multivariate number of features
        tr_tst - train-test ratio
        '''
        self.check_infinite_values(self.df)

        dfY = self.df["PX_OPEN"].copy()
        dfX = self.df[["PX_OPEN", "PX_LAST", "RSI_14D", "PX_TREND", "PX_VTREND", "TWEET_POSTIVIE", "TWEET_NEGATIVE",
                   "NEWS_POSITIVE", "NEWS_NEGATIVE", "VOLATILITY", "MOMENTUM"]].copy()
        idx = dfX.index[(mwin - 1):]

        mX, cols = self.prepare_multivariate_data(dfX, mwin, m_ftrs, idx)

        for ahead in self._lahead:
            mXl = mX[:-(ahead + 1), :, :]
            mYl = dfY.iloc[ahead + mwin:]
            avmX = self.calculate_mean(mXl, axis=1)

            avmXc = mXl - avmX[:, None, :]
            pavX = pd.DataFrame(avmX, columns=dfX.columns)
            pavX.set_index(idx[ahead+1:], drop=True, inplace=True)
            mxmX = np.round(self.calculate_max(avmXc, axis=(0, 1)), 2)
            mnmX = np.round(self.calculate_min(avmXc, axis=(0, 1)), 2)
            mvdd = {"mean": pavX, "min": mnmX, "max": mxmX}
            mXn = (avmXc - mnmX[None, None, :]) / (mxmX[None, None, :] - mnmX[None, None, :] + 0.00001)
            mYn = ((mYl.to_numpy() - avmX[:, 0]) - mnmX[0]) / (mxmX[0] - mnmX[0] + 0.00001)

            mXn = mXn.astype("float32")
            mYn = mYn.astype("float32")

            mtrainX, mtrainY, mtestX, mtestY, pmod = self.prepare_data_for_modeling(mXn, mYn, multi=True)

            self.check_nan_values(mtrainX, mtestX, ahead)

            xdx = idx[:-(ahead + 1)]
            self.mserial_dict[ahead] = {"x": mXl, "y": mYl, "nx": mXn, "ny": mYn, "numt": pmod,
                                        "trainX": mtrainX, "trainY": mtrainY,
                                        "testX": mtestX, "testY": mtestY, "vdd": mvdd, "cnms": cols,
                                        "idtest": xdx[pmod:]}

    def prepare_multivariate_data(self, dfX, win, n_ftrs, idx):
        '''
        Returns the multivariate data for modeling
        
        Parameters:
        dfX - multivariate features data
        win - window size
        n_ftrs - number of features
        idx - index of the data
        '''
        mX = []
        cols = dfX.columns
        for i in range(len(dfX.columns)):
            lX = np.lib.stride_tricks.sliding_window_view(dfX.iloc[:, i].to_numpy(), window_shape=win)[::n_ftrs]
            ss = pd.DataFrame.from_records(lX)
            ss.set_index(idx, drop=True, inplace=True)
            mX.append(ss.to_numpy())
        mX = np.transpose(np.stack(mX, axis=1), (0, 2, 1))
        return mX, cols

    def compute_sentiment_scores(self):
        '''
        Computes the sentiment scores for the data
        '''
        self.df['TWEET_POSTIVIE'] = self.df['TWITTER_POS_SENTIMENT_COUNT'] / self.df['TWITTER_PUBLICATION_COUNT'].replace(0, 1)
        self.df['TWEET_NEGATIVE'] = self.df['TWITTER_NEG_SENTIMENT_COUNT'] / self.df['TWITTER_PUBLICATION_COUNT'].replace(0, 1)
        self.df['NEWS_POSITIVE'] = self.df['NEWS_POS_SENTIMENT_COUNT'] / self.df['NEWS_PUBLICATION_COUNT'].replace(0, 1)
        self.df['NEWS_NEGATIVE'] = self.df['NEWS_NEG_SENTIMENT_COUNT'] / self.df['NEWS_PUBLICATION_COUNT'].replace(0, 1)

    def compute_volatility(self):
        '''
        Computes the volatility for the data with the Garman and Klass model
        '''
        self.df['VOLATILITY'] = 1/2 * (np.log(self.df["PX_HIGH"]) - np.log(self.df["PX_LOW"]))**2 - (2 * np.log(2) - 1) * (np.log(self.df["PX_LAST"]) - np.log(self.df["PX_OPEN"]))**2

    def compute_momentum(self):
        '''
        Computes the momentum for the data
        '''
        self.df['MOMENTUM'] = self.df['PX_LAST'] - self.df['PX_LAST'].shift(1)

    def compute_trend(self):
        '''
        Computes the trend for the price and volume data
        '''
        self.df['PX_TREND'] = 2 * (self.df['PX_LAST'] - self.df['PX_OPEN']) / (self.df['PX_OPEN'] + self.df['PX_LAST'])
        self.df['PX_VTREND'] = self.df['PX_TREND'] * self.df['VOLUME']

    def check_nan_values(self, trainX, testX, ahead):
        '''
        Checks for NaN values in the data
        '''
        if np.isinf(trainX).any() or np.isnan(trainX).any():
            print("Stock {} ahead: {} has nans.".format(self.ticker, ahead))
            print(trainX)

        if np.isnan(testX).any():
            print("Stock {} ahead: {} has nans.".format(self.ticker, ahead))
            print(testX)

    def check_infinite_values(self, df):
        '''
        Checks for infinite values in the data
        '''
        if np.isinf(df).values.sum() > 0:
            import pdb
            pdb.set_trace()

    def prepare_data_for_modeling(self, Xn, Yn, multi):
        '''
        Returns the data for modeling
        
        Parameters:
        Xn - normalized features data
        Yn - normalized target data
        multi - boolean value for multivariate data'''
        pmod = int(Xn.shape[0] * self._tr_tst)
        if multi == False:  # Univariate
            trainX = Xn.iloc[:pmod,:]
            trainY = Yn.iloc[:pmod]
            testX  = Xn.iloc[pmod:,:]
            testY  = Yn.iloc[pmod:]
        else:  # Multivariate
            trainX = Xn[:pmod, :, :]
            trainY = Yn[:pmod]
            testX = Xn[pmod:, :, :]
            testY = Yn[pmod:]
        return trainX, trainY, testX, testY, pmod
    
    def process_stocks(self):
        '''
        Processes the stock data
        '''
        self.compute_sentiment_scores()
        self.compute_volatility()
        self.compute_momentum()
        self.compute_trend()
        self.df.dropna(inplace=True)

    def calculate_mean(self, data, axis=None):
        return np.mean(data, axis=axis)

    def calculate_min(self, data, axis=None):
        return np.min(data, axis=axis)

    def calculate_max(self, data, axis=None):      
        return np.max(data, axis=axis)


def save_data(fich, out_path, lahead, lpar, tot_res):
    '''
    Saves the data to a pickle file
    '''
    with open(fich, 'wb') as file:
            pickle.dump(out_path, file)
            pickle.dump(fich, file)
            pickle.dump(lahead, file)
            pickle.dump(lpar, file)
            pickle.dump(tot_res, file)
    file.close()


# from https://stackoverflow.com/questions/6076690/verbose-level-with-argparse-and-multiple-v-options
class VAction(argparse.Action):
    '''
    Custom action class to handle the verbose option
    '''

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(VAction, self).__init__(option_strings, dest, nargs, const,
                                      default, type, choices, required,
                                      help, metavar)
        self.values = 0
    def __call__(self, parser, args, values, option_string=None):
        # print('values: {v!r}'.format(v=values))
        if values is None:
            self.values += 1
        else:
            try:
                self.values = int(values)
            except ValueError:
                self.values = values.count('v')+1
        setattr(args, self.dest, self.values)      


def main(args):
    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)

    if args.verbose > 0:
        print("Additional Info:")
        # Add additional information here
        print("Processing data with the configuration file:", args.params_file)

    # Load configuration parameters from the YAML file
    data_path = config['data']['data_path']
    out_path = config['data']['output_path']
    tickers_list = config['tickers']
    filename_structure = config['data']['filename_structure']

    # stock_list = ["AAPL", "ADBE", "AMZN", "AVGO", "CMCSA", "COST", "CSCO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "PEP", "TMUS", "TSLA"]
    for ticker in tickers_list:

        filename = filename_structure.format(ticker=ticker, date=args.date)
        file = os.path.join(data_path, filename)
        assert os.path.exists(file), f"El archivo {file} no existe."

        scen = args.scenario
        for scenario in config['serialization']['scenarios']:
            if scenario['name'] == f'scenario_{scen}':
                win = scenario['win']
                tr_tst = scenario['tr_tst']
                lahead = scenario['lahead']
                n_ftrs = scenario['n_features']
                break

        stock = Stock(ticker, file, lahead, tr_tst)
        stock.process_stocks()

        # Univariate data processing

        stock.process_univariate_data(win)
        fdat1 = os.path.join(out_path, "{:02}/{:03}-input-output.pkl".format(win, ticker))
        lpar  = [win, tr_tst]

        # Save univariate data
        if os.path.exists(fdat1):
            save_data(fdat1, out_path, lahead, lpar, stock.serial_dict)
        else:
            directory = os.path.dirname(fdat1)
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory {directory} created.")

            save_data(fdat1, out_path, lahead, lpar, stock.serial_dict)                
            print(f"File {fdat1} created and data saved.")

        # Multivariate data processing
        stock.process_multivariate_data(win, n_ftrs)
        fdat2 = os.path.join(out_path, "{:02}/{:03}-m-input-output.pkl".format(win, ticker))
        lpar = [win, n_ftrs, tr_tst]

        # Save multivariate data
        if os.path.dirname(fdat2):
            save_data(fdat2, out_path, lahead, lpar, stock.mserial_dict)
        else:
            directory = os.path.dirname(fdat2)
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory {directory} created.")

            save_data(fdat2, out_path, lahead, lpar, stock.mserial_dict)
            print(f"File {fdat2} created and data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    parser.add_argument("--date", help="Date for filename structure")
    parser.add_argument("--scenario", help="Number of scenario to process", type=int)
    args = parser.parse_args()
    main(args)