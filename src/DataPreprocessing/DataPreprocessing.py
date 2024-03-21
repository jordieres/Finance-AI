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
    def prepare_data_for_modeling(mXn, mYn, tr_tst):
        pmod = int(mXn.shape[0] * tr_tst)
        mtrainX = mXn[:pmod, :, :]
        mtrainY = mYn[:pmod]
        mtestX = mXn[pmod:, :, :]
        mtestY = mYn[pmod:]
        return mtrainX, mtrainY, mtestX, mtestY, pmod

    @staticmethod
    def check_nan_values(mtrainX, mtestX, stock, ahead):
        if np.isinf(mtrainX).any() or np.isnan(mtrainX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(mtrainX)
            import pdb
            pdb.set_trace()
        if np.isnan(mtestX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(mtestX)

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
    def prepare_serial_dict(mXl, mYl, mXn, mYn, pmod, mvdd, mtrainX, mtrainY, mtestX, mtestY, cols, xdx, ahead):
        mserial_dict = {
            "x": mXl, "y": mYl, "nx": mXn, "ny": mYn, "numt": pmod,
            "vdd": mvdd, "trX": mtrainX, "trY": mtrainY,
            "tsX": mtestX, "tsY": mtestY, "cnms": cols,
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
    
    # Parametros
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")

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

    for stock in data_processor.stock_list:
        mserial_dict = {}
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

            mtrainX, mtrainY, mtestX, mtestY, pmod = data_manipulation.prepare_data_for_modeling(mXn, mYn, tr_tst)

            data_manipulation.check_nan_values(mtrainX, mtestX, stock, ahead)

            xdx = idx[:-(ahead+1)]
            mserial_dict[ahead] = data_manipulation.prepare_serial_dict(mXl, mYl, mXn, mYn, pmod, mvdd, mtrainX, mtrainY, mtestX, mtestY, cols, xdx, ahead)

        fdat2 = os.path.join(path, "{:02}/m-input-output.pkl".format(mwin))
        lpar = [mwin, mdeep, m_ftrs, tr_tst]

        with open(fdat2, 'wb') as file:
            pickle.dump(path, file)
            pickle.dump(fdat2, file)
            pickle.dump(lahead, file)
            pickle.dump(lpar, file)
            pickle.dump(data_processor.stock_list, file)
            pickle.dump(mserial_dict, file)
        file.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)
