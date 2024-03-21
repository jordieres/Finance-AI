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
    def __init__(self, path="../Datasets/", stock_list=None):
        self.path = path
        self.stock_list = stock_list if stock_list else ["AAPL", "ADBE", "AMZN", "AVGO", "CMCSA", "COST", "CSCO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "PEP", "TMUS", "TSLA"]
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
    
def main(args):
    with open(args.params_file, 'r') as f:
        params = yaml.safe_load(f)

    warnings.filterwarnings('ignore')
    mpl.rcParams['figure.figsize'] = (18, 10)
    mpl.rcParams['axes.grid'] = False
    sns.set_style("whitegrid")

    data_processor = DataProcessor(params['data_path'])
    data_processor.load_data()

    data_manipulation = DataManipulation()

    path = params['output_path']
    mwin = params['mwin']
    mdeep = params['mdeep']
    m_ftrs = params['m_ftrs']
    tr_tst = params['tr_tst']
    lahead = [1, 7, 14, 30, 90]
    n_ftrs = params['n_ftrs']

    multivariate_serial_dict = {}
    single_variable_serial_dict = {}
    for stock in data_processor.stock_list:
        multivariate_serial_dict[stock] = {}
        single_variable_serial_dict[stock] = {}
        for ahead in lahead:
            df = data_processor.df_dict[stock].copy()

            # Multivariate data processing
            win_x = np.lib.stride_tricks.sliding_window_view(df.PX_OPEN.to_numpy(), window_shape=mwin)[::1]

            Y = df.iloc[ahead + mwin:]['PX_OPEN']
            X = pd.DataFrame.from_records(win_x)

            X = X.iloc[:-(ahead + 1), :]
            X.set_index(df.index[(mwin - 1):-(ahead + 1)], drop=True, inplace=True)

            xm = X.mean(axis=1)
            cX = X.sub(X.mean(axis=1), axis=0)

            mnx = round(min(cX.min()))  # whole min
            mxx = round(max(cX.max()))  # whole max
            vdd = pd.DataFrame({'mean': xm, 'min': mnx, 'max': mxx})  # DF with ranges
            vdd.set_index(X.index, drop=True, inplace=True)

            cXn = cX.apply(lambda x: (x - mnx) / (mxx - mnx), axis=1)
            cYn = pd.Series([((i - j) - mnx) / (mxx - mnx) for i, j in zip(Y.tolist(), xm.tolist())],
                            index=Y.index)
            cXn = cXn.astype('float32')
            cYn = cYn.astype('float32')

            pmod = int(cXn.shape[0] * tr_tst)
            trainX = cXn.iloc[:pmod, :]
            trainY = cYn.iloc[:pmod]
            testX = cXn.iloc[pmod:, :]
            testY = cYn.iloc[pmod:]

            multivariate_serial_dict[stock][ahead] = {'x': X, 'y': Y, 'nx': cXn, 'ny': cYn, 'numt': pmod, 'vdd': vdd,
                                                      'trX': trainX, 'trY': trainY, 'tsX': testX, 'tsY': testY}

            # Single variable data processing
            win_x_single = np.lib.stride_tricks.sliding_window_view(df.PX_OPEN.to_numpy(), window_shape=mwin)[::1]

            Y_single = df.iloc[ahead + mwin:]['PX_OPEN']
            X_single = pd.DataFrame.from_records(win_x_single)

            X_single = X_single.iloc[:-(ahead + 1), :]
            X_single.set_index(df.index[(mwin - 1):-(ahead + 1)], drop=True, inplace=True)

            xm_single = X_single.mean(axis=1)
            cX_single = X_single.sub(X_single.mean(axis=1), axis=0)

            mnx_single = round(min(cX_single.min()))  # whole min
            mxx_single = round(max(cX_single.max()))  # whole max
            vdd_single = pd.DataFrame({'mean': xm_single, 'min': mnx_single, 'max': mxx_single})  # DF with ranges
            vdd_single.set_index(X_single.index, drop=True, inplace=True)

            cXn_single = cX_single.apply(lambda x: (x - mnx_single) / (mxx_single - mnx_single), axis=1)
            cYn_single = pd.Series([((i - j) - mnx_single) / (mxx_single - mnx_single) for i, j in
                                     zip(Y_single.tolist(), xm_single.tolist())], index=Y_single.index)
            cXn_single = cXn_single.astype('float32')
            cYn_single = cYn_single.astype('float32')

            pmod_single = int(cXn_single.shape[0] * tr_tst)
            trainX_single = cXn_single.iloc[:pmod_single, :]
            trainY_single = cYn_single.iloc[:pmod_single]
            testX_single = cXn_single.iloc[pmod_single:, :]
            testY_single = cYn_single.iloc[pmod_single:]

            single_variable_serial_dict[stock][ahead] = {'x': X_single, 'y': Y_single, 'nx': cXn_single, 'ny': cYn_single,
                                                         'numt': pmod_single, 'vdd': vdd_single,
                                                         'trX': trainX_single, 'trY': trainY_single, 'tsX': testX_single,
                                                         'tsY': testY_single}

    # Guardar los resultados
    fdat1 = os.path.join(path, "{:02}/input-output.pkl".format(mwin))
    fdat2 = os.path.join(path, "{:02}/m-input-output.pkl".format(mwin))
    lpar = [mwin, mdeep, n_ftrs, tr_tst]

    with open(fdat1, 'wb') as file:
        pickle.dump(path, file)
        pickle.dump(fdat1, file)
        pickle.dump(lahead, file)
        pickle.dump(lpar, file)
        pickle.dump(data_processor.stock_list, file)
        pickle.dump(single_variable_serial_dict, file)

    with open(fdat2, 'wb') as file:
        pickle.dump(path, file)
        pickle.dump(fdat2, file)
        pickle.dump(lahead, file)
        pickle.dump(lpar, file)
        pickle.dump(data_processor.stock_list, file)
        pickle.dump(multivariate_serial_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("params_preprocessing.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)

