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



class DataProcessor:
    def __init__(self):
        self.path = "../../Datasets/"
        self.stock_list= ["AAPL", "ADBE", "AMZN", "AVGO", "CMCSA", "COST", "CSCO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "PEP", "TMUS", "TSLA"]
        self.df_dict = {}
        self.tot_res = {}

    def load_data(self):
        for stock in self.stock_list:
            fich = os.path.join(self.path, f"{stock} US Equity_060124.csv")
            df = pd.read_csv(fich, sep=",", index_col=0, parse_dates=True)
            df['PX_TREND'] = 2 * (df['PX_LAST'] - df['PX_OPEN']) / (df['PX_OPEN'] + df['PX_LAST'])
            df['PX_VTREND'] = df['PX_TREND'] * df['VOLUME']
            nans = sum(df['PX_OPEN'].isna()) * 100 / df.shape[0]
            df = df.dropna()
            print('{}: NaNs are about {}% in the original dataset. Size: {}'.format(stock, str(round(nans)), str(df.shape)))
            self.df_dict[stock] = df
            self.tot_res['INP'] = self.df_dict

class DataManipulation:
    @staticmethod
    def compute_sentiment_scores(df, sentiment_count_col, publication_count_col):
        df[publication_count_col].replace(to_replace=0, value=1, inplace=True)
        sentiment_score = df[sentiment_count_col] / df[publication_count_col]
        return sentiment_score

    @staticmethod
    def compute_volatility(df):
        volatility = 1/2 * (np.log(df["PX_HIGH"]) - np.log(df["PX_LOW"]))**2 - (2 * np.log(2) - 1) * (np.log(df["PX_LAST"]) - np.log(df["PX_OPEN"]))**2
        return volatility

    @staticmethod
    def check_infinite_values(df):
        if np.isinf(df).values.sum() > 0:
            import pdb
            pdb.set_trace()

    @staticmethod
    def windowing(dfX, win, nvar, idx):
        lX = np.lib.stride_tricks.sliding_window_view(dfX.to_numpy(), window_shape=win)[::nvar]
        mX = pd.DataFrame.from_records(lX)
        mX.set_index(idx, drop=True, inplace=True)
        return mX

    @staticmethod
    def normalize_data(mXl, avmX, mYl, dfX, idx, ahead):
        avmXc = mXl - avmX[:, None, :]
        pavX = pd.DataFrame(np.mean(mXl, axis=1), columns=dfX.columns)
        pavX.set_index(idx[ahead+1:], drop=True, inplace=True)
        mxmX = np.round(np.max(avmXc, axis=(0, 1)), 2)
        mnmX = np.round(np.min(avmXc, axis=(0, 1)), 2)
        mvdd = {"mean": pavX, "min": mnmX, "max": mxmX}
        mXn = (avmXc - mnmX[None, None, :]) / (mxmX[None, None, :] - mnmX[None, None, :] + 0.00001)
        mYn = ((mYl.to_numpy() - avmX[:, 0]) - mnmX[0]) / (mxmX[0] - mnmX[0] + 0.00001)
        return mXn, mYn, mvdd

    @staticmethod
    def prepare_data_for_modeling(Xn, Yn, tr_tst, multi=False):
        pmod = int(Xn.shape[0] * tr_tst)
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

    @staticmethod
    def check_nan_values(trainX, testX, stock, ahead):
        if np.isinf(trainX).any() or np.isnan(trainX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(trainX)
            import pdb
            pdb.set_trace()
        if np.isnan(testX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(testX)

    @staticmethod
    def prepare_multivariate_data(dfX, mwin, m_ftrs, idx):
        mX = []
        cols = dfX.columns
        for i in range(len(dfX.columns)):
            ss = DataManipulation.windowing(dfX.iloc[:, i], mwin, m_ftrs, idx)
            mX.append(ss.to_numpy())
        mX = np.transpose(np.stack(mX, axis=1), (0, 2, 1))
        return mX, cols

    @staticmethod
    def prepare_serial_dict(x, y, Xn, Yn, pmod, vdd, trainX, trainY, testX, testY, cols=None, xdx=None):
        if cols is None and xdx is None:
            serial_dict = {
                "x": x, "y": y, "nx": Xn, "ny": Yn, "numt": pmod,
                "vdd": vdd, "trX": trainX, "trY": trainY,
                "tsX": testX, "tsY": testY
            }
            return serial_dict

        else:
            mserial_dict = {
                "x": x, "y": y, "nx": Xn, "ny": Yn, "numt": pmod,
                "vdd": vdd, "trX": trainX, "trY": trainY,
                "tsX": testX, "tsY": testY, "cnms": cols,
                "idtest": xdx[pmod:]
            }
            return mserial_dict
    
# from https://stackoverflow.com/questions/6076690/verbose-level-with-argparse-and-multiple-v-options
class VAction(argparse.Action):
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
#


def main(args):

    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)

    if args.verbose > 0:
        print("Additional Info:")
        # Add additional information here
        print("Processing data with the configuration file:", args.params_file)
    
    warnings.filterwarnings('ignore')
    mpl.rcParams['figure.figsize'] = (18, 10)
    mpl.rcParams['axes.grid'] = False
    sns.set_style("whitegrid")

    data_processor = DataProcessor()
    data_processor.load_data()

    data_manipulation = DataManipulation()

    path = config['output_path']
    mwin = config['mwin']
    mdeep = config['mdeep']
    m_ftrs = config['m_ftrs']
    tr_tst = config['tr_tst']
    lahead = [1, 7, 14, 30, 90]

    win = config['win']
    deep = config['deep']
    n_ftrs = config['n_ftrs']

# Univariate data preparation
    
    serial_dict = {}
    for stock in data_processor.stock_list:

        serial_dict[stock] = {}
        for ahead in lahead:
            df = data_processor.df_dict[stock].copy()

            # Window of relevant data
            win_x = np.lib.stride_tricks.sliding_window_view(df.PX_OPEN.to_numpy(), window_shape = win)[::1]

            # Shift the Y axis
            Y= df.iloc[ahead+win:]['PX_OPEN']
            X = pd.DataFrame.from_records(win_x)

            # Cutting out records with no data in Y+ahead respecting the win+ahead offset
            X = X.iloc[:-(ahead+1),:] # Skipping the gap in forecasting values
            X.set_index(df.index[(win-1):-(ahead+1)],drop=True, inplace=True)

            xm = X.mean(axis=1)
            cX = X.sub(X.mean(axis=1), axis=0)

            #  Scaling and storing ranges in vdd
            mnx= round(min(cX.min())) # whole min
            mxx= round(max(cX.max())) # whole max
            vdd= pd.DataFrame({'mean':xm,'min':mnx,'max':mxx}) # DF with ranges
            vdd.set_index(X.index,drop=True,inplace=True)

            # Normalizing X and Y
            cXn= cX.apply(lambda x: (x-mnx)/(mxx-mnx), axis=1)
            cYn= pd.Series([((i-j)-mnx)/(mxx-mnx) for i,j in zip(Y.tolist(),xm.tolist())], index=Y.index)
            cXn = cXn.astype('float32')
            cYn = cYn.astype('float32')

            # Data set preparation for modeling, starting from cXn and cYn
            trainX, trainY, testX, testY, pmod = data_manipulation.prepare_data_for_modeling(cXn, cYn, tr_tst)
            

            serial_dict[stock][ahead] = data_manipulation.prepare_serial_dict(X, Y, cXn, cYn, pmod, vdd, trainX, trainY, testX, testY)
            
            data_processor.tot_res["INP"] = serial_dict
            
        fdat1 = "../../DataProcessed/{:02}/input-output.pkl".format(win)
        lpar  = [win, deep, n_ftrs,tr_tst]

        with open(fdat1, 'wb') as file:
            pickle.dump(path, file)
            pickle.dump(fdat1,file)
            pickle.dump(lahead,file)
            pickle.dump(lpar,file)
            pickle.dump(data_processor.stock_list,file)
            pickle.dump(data_processor.tot_res, file)
        file.close()
                                                    

# Multivariate data preparation
        
    mserial_dict = {}
    for stock in data_processor.stock_list:
        
        mserial_dict[stock] = {}
        df = data_processor.df_dict[stock].copy()
        mdf = df[["PX_OPEN", "PX_LAST", "RSI_14D", "PX_TREND", "PX_VTREND"]]
        mdf.loc[:, "TWEETPR"] = data_manipulation.compute_sentiment_scores(df, "TWITTER_POS_SENTIMENT_COUNT", "TWITTER_PUBLICATION_COUNT")
        mdf.loc[:, "TWEETNR"] = data_manipulation.compute_sentiment_scores(df, "TWITTER_NEG_SENTIMENT_COUNT", "TWITTER_PUBLICATION_COUNT")
        mdf.loc[:, "NEWSPR"] = data_manipulation.compute_sentiment_scores(df, "NEWS_POS_SENTIMENT_COUNT", "NEWS_PUBLICATION_COUNT")
        mdf.loc[:, "NEWSNR"] = data_manipulation.compute_sentiment_scores(df, "NEWS_NEG_SENTIMENT_COUNT", "NEWS_PUBLICATION_COUNT")
        mdf.loc[:, "VOLATILITY"] = data_manipulation.compute_volatility(df)
        data_manipulation.check_infinite_values(mdf)

        dfY = mdf["PX_OPEN"].copy()
        dfX = mdf[["PX_OPEN", "PX_LAST", "RSI_14D", "PX_TREND", "PX_VTREND", "TWEETPR", "TWEETNR", "NEWSPR", "NEWSNR", "VOLATILITY"]].copy()
        idx = dfX.index[(mwin-1):]

        mX, cols = data_manipulation.prepare_multivariate_data(dfX, mwin, m_ftrs, idx)

        for ahead in lahead:
            mXl = mX[:-(ahead+1), :, :]
            mYl = dfY.iloc[ahead+mwin:]
            avmX = np.mean(mXl, axis=1)

            mXn, mYn, mvdd = data_manipulation.normalize_data(mXl, avmX, mYl, dfX, idx, ahead)

            mXn = mXn.astype("float32")
            mYn = mYn.astype("float32")

            mtrainX, mtrainY, mtestX, mtestY, pmod = data_manipulation.prepare_data_for_modeling(mXn, mYn, tr_tst, multi=True)

            data_manipulation.check_nan_values(mtrainX, mtestX, stock, ahead)

            xdx = idx[:-(ahead+1)]
            mserial_dict[stock][ahead] = data_manipulation.prepare_serial_dict(mXl, mYl, mXn, mYn, pmod, mvdd, mtrainX, mtrainY, mtestX, mtestY, cols, xdx)

        
        data_processor.tot_res["INP_MSERIAL"] = mserial_dict

        fdat2 = os.path.join(path, "{:02}/m-input-output.pkl".format(mwin))
        lpar = [mwin, mdeep, m_ftrs, tr_tst]

        with open(fdat2, 'wb') as file:
            pickle.dump(path, file)
            pickle.dump(fdat2, file)
            pickle.dump(lahead, file)
            pickle.dump(lpar, file)
            pickle.dump(data_processor.stock_list, file)
            pickle.dump(data_processor.tot_res, file)
        file.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)
