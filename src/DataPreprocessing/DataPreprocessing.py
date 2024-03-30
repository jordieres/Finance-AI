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


class DataManipulation:
    @staticmethod
    def compute_sentiment_scores(df, sentiment_count_col, publication_count_col):
        '''Returns the sentiment scores
        
        Arguments:
        df - data
        sentiment_count_col - sentiment count column
        publication_count_col - publication count column
        '''

        df[publication_count_col].replace(to_replace=0, value=1, inplace=True)
        sentiment_score = df[sentiment_count_col] / df[publication_count_col]
        return sentiment_score

    @staticmethod
    def compute_volatility(df): # compute volatility of the stock with the Garman and Klass model
        '''Returns the volatility of the stock
        
        Arguments:
        df - data
        '''

        volatility = 1/2 * (np.log(df["PX_HIGH"]) - np.log(df["PX_LOW"]))**2 - (2 * np.log(2) - 1) * (np.log(df["PX_LAST"]) - np.log(df["PX_OPEN"]))**2
        return volatility

    @staticmethod
    def check_infinite_values(df): # check for infinite values in the data
        '''Arguments:
        df - data
        '''

        if np.isinf(df).values.sum() > 0:
            import pdb
            pdb.set_trace()

    @staticmethod
    def windowing(dfX, win, nvar, idx):
        '''Returns the windowed data
        
        Arguments:
        dfX - data
        win - window size
        nvar - number of variables
        idx - index of the data
        '''

        lX = np.lib.stride_tricks.sliding_window_view(dfX.to_numpy(), window_shape=win)[::nvar]
        mX = pd.DataFrame.from_records(lX)
        mX.set_index(idx, drop=True, inplace=True)
        return mX

    @staticmethod
    def normalize_data(mXl, avmX, mYl, dfX, idx, ahead):
        '''Returns the normalized data and the mean, min, and max values of the data
        
        Arguments:
        mXl - multivariate data
        avmX - average of the multivariate data
        mYl - target data
        dfX - multivariate data
        idx - index of the data
        ahead - ahead value
        '''

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
        '''Returns the training and testing data for modeling
        
        Arguments:
        Xn - normalized X data
        Yn - normalized Y data
        tr_tst - train-test ratio
        multi - boolean value to indicate if the data is multivariate or not
        '''

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
    def check_nan_values(trainX, testX, stock, ahead): # check for nans in the data and print the number of nans
        '''
        Arguments:
        trainX - training data
        testX - testing data
        stock - stock name
        ahead - ahead value
        '''

        if np.isinf(trainX).any() or np.isnan(trainX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(trainX)
            import pdb
            pdb.set_trace()
        if np.isnan(testX).any():
            print("Stock {} ahead: {} has nans.".format(stock, ahead))
            print(testX)

    @staticmethod
    def prepare_multivariate_data(dfX, mwin, m_ftrs, idx): # prepare multivariate data for modeling
        '''
        Returns the multivariate data and the columns of the data

        Arguments:
        dfX - multivariate data
        mwin - multivariate window size
        m_ftrs - number of features
        idx - index of the data
        '''

        mX = []
        cols = dfX.columns
        for i in range(len(dfX.columns)):
            ss = DataManipulation.windowing(dfX.iloc[:, i], mwin, m_ftrs, idx)
            mX.append(ss.to_numpy())
        mX = np.transpose(np.stack(mX, axis=1), (0, 2, 1))
        return mX, cols

    @staticmethod
    def prepare_serial_dict(x, y, Xn, Yn, pmod, vdd, trainX, trainY, testX, testY, cols=None, xdx=None):
        '''
        Returns a dictionary with the following keys: x, y, nx, ny, numt, vdd, trX, trY, tsX, tsY, cnms, idtest

        Arguments:
        x - original X data
        y - original Y data
        Xn - normalized X data
        Yn - normalized Y data
        pmod - number of training samples
        vdd - dictionary with mean, min, and max values
        trainX - training X data
        trainY - training Y data
        testX - testing X data
        testY - testing Y data
        cols - columns of the multivariate data
        xdx - index of the testing data
        '''

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
        

class DataProcessor(DataManipulation): # DataProcessor class inherits from DataManipulation class
    def __init__(self, data_path, out_path):
        '''Arguments:
        data_path - path to the data files
        out_path - path to the output files
        '''

        self.path = data_path
        self.out_path = out_path
        self.stock_list= ["AAPL", "ADBE", "AMZN", "AVGO", "CMCSA", "COST", "CSCO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "PEP", "TMUS", "TSLA"]
        self.lahead = [1, 7, 14, 30, 90] # list of ahead values
        self.df_dict = {}
        self.tot_res = {}
        
        self.serial_dict = {}
        self.mserial_dict = {}

    def load_data(self): # load data from the csv files
        for stock in self.stock_list:
            fich = os.path.join(self.path, f"{stock} US Equity_060124.csv")
            assert os.path.exists(fich), f"El archivo {fich} no existe."
            df = pd.read_csv(fich, sep=",", index_col=0, parse_dates=True)
            df['PX_TREND'] = 2 * (df['PX_LAST'] - df['PX_OPEN']) / (df['PX_OPEN'] + df['PX_LAST'])
            df['PX_VTREND'] = df['PX_TREND'] * df['VOLUME']
            nans = sum(df['PX_OPEN'].isna()) * 100 / df.shape[0]
            df = df.dropna()
            print('{}: NaNs are about {}% in the original dataset. Size: {}'.format(stock, str(round(nans)), str(df.shape)))
            self.df_dict[stock] = df
            self.tot_res['INP'] = self.df_dict

    def save_data(self, fdat, lpar, stock_list, tot_res): # save data to a pickle file
        '''
        Arguments:
        fdat - file name
        lpar - list of parameters
        stock_list - list of stocks
        tot_res - total results
        '''

        with open(fdat, 'wb') as file:
            pickle.dump(self.out_path, file)
            pickle.dump(fdat, file)
            pickle.dump(self.lahead, file)
            pickle.dump(lpar, file)
            pickle.dump(stock_list, file)
            pickle.dump(tot_res, file)
        file.close()

    def process_univariate_data(self, win, tr_tst):
        '''
        Arguments:
        win - window size
        tr_tst - train-test ratio
        '''

        for stock in self.stock_list:
            self.serial_dict[stock] = {}
            for ahead in self.lahead:
                df = self.df_dict[stock].copy()

                win_x = np.lib.stride_tricks.sliding_window_view(df.PX_OPEN.to_numpy(), window_shape = win)[::1]

                Y= df.iloc[ahead+win:]['PX_OPEN']
                X = pd.DataFrame.from_records(win_x)

                X = X.iloc[:-(ahead+1),:]
                X.set_index(df.index[(win-1):-(ahead+1)],drop=True, inplace=True)

                xm = X.mean(axis=1)
                cX = X.sub(X.mean(axis=1), axis=0)

                mnx= round(min(cX.min()))
                mxx= round(max(cX.max()))
                vdd= pd.DataFrame({'mean':xm,'min':mnx,'max':mxx})
                vdd.set_index(X.index,drop=True,inplace=True)

                cXn= cX.apply(lambda x: (x-mnx)/(mxx-mnx), axis=1)
                cYn= pd.Series([((i-j)-mnx)/(mxx-mnx) for i,j in zip(Y.tolist(),xm.tolist())], index=Y.index)
                cXn = cXn.astype('float32')
                cYn = cYn.astype('float32')

                trainX, trainY, testX, testY, pmod = self.prepare_data_for_modeling(cXn, cYn, tr_tst)

                self.serial_dict[stock][ahead] = self.prepare_serial_dict(X, Y, cXn, cYn, pmod, vdd, trainX, trainY, testX, testY)
            
            self.tot_res["INP"] = self.serial_dict

    def process_multivariate_data(self, mwin, m_ftrs, tr_tst):
        '''
        Arguments:
        mwin - multivariate window size
        m_ftrs - multivariate number of features
        tr_tst - train-test ratio
        '''
        
        for stock in self.stock_list:
            self.mserial_dict[stock] = {}
            df = self.df_dict[stock].copy()
            mdf = df[["PX_OPEN", "PX_LAST", "RSI_14D", "PX_TREND", "PX_VTREND"]]
            mdf.loc[:, "TWEETPR"] = self.compute_sentiment_scores(df, "TWITTER_POS_SENTIMENT_COUNT", "TWITTER_PUBLICATION_COUNT")
            mdf.loc[:, "TWEETNR"] = self.compute_sentiment_scores(df, "TWITTER_NEG_SENTIMENT_COUNT", "TWITTER_PUBLICATION_COUNT")
            mdf.loc[:, "NEWSPR"] = self.compute_sentiment_scores(df, "NEWS_POS_SENTIMENT_COUNT", "NEWS_PUBLICATION_COUNT")
            mdf.loc[:, "NEWSNR"] = self.compute_sentiment_scores(df, "NEWS_NEG_SENTIMENT_COUNT", "NEWS_PUBLICATION_COUNT")
            mdf.loc[:, "VOLATILITY"] = self.compute_volatility(df)
            self.check_infinite_values(mdf)

            dfY = mdf["PX_OPEN"].copy()
            dfX = mdf[["PX_OPEN", "PX_LAST", "RSI_14D", "PX_TREND", "PX_VTREND", "TWEETPR", "TWEETNR", "NEWSPR", "NEWSNR", "VOLATILITY"]].copy()
            idx = dfX.index[(mwin-1):]

            mX, cols = self.prepare_multivariate_data(dfX, mwin, m_ftrs, idx)

            for ahead in self.lahead:
                mXl = mX[:-(ahead+1), :, :]
                mYl = dfY.iloc[ahead+mwin:]
                avmX = np.mean(mXl, axis=1)

                mXn, mYn, mvdd = self.normalize_data(mXl, avmX, mYl, dfX, idx, ahead)

                mXn = mXn.astype("float32")
                mYn = mYn.astype("float32")

                mtrainX, mtrainY, mtestX, mtestY, pmod = self.prepare_data_for_modeling(mXn, mYn, tr_tst, multi=True)

                self.check_nan_values(mtrainX, mtestX, stock, ahead)

                xdx = idx[:-(ahead+1)]
                self.mserial_dict[stock][ahead] = self.prepare_serial_dict(mXl, mYl, mXn, mYn, pmod, mvdd, mtrainX, mtrainY, mtestX, mtestY, cols, xdx)

            self.tot_res["INP_MSERIAL"] = self.mserial_dict

    
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

def load_preprocessed_data(path, win, multi=False):
    '''Returns the preprocessed data as a list of objects
    
    Arguments:
    path - path to the preprocessed data
    win - window size
    multi - boolean value to indicate if the data is multivariate or not
    '''

    if multi == False:
        fdat = os.path.join(path, "{:02}/input-output.pkl".format(win))

    else:
        fdat = os.path.join(path, "{:02}/m-input-output.pkl".format(win))

    objects = []
    with (open(fdat, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

def main(args):

    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    f.close()

    if args.verbose > 0:
        print("Additional Info:")
        # Add additional information here
        print("Processing data with the configuration file:", args.params_file)
    
    warnings.filterwarnings('ignore')
    mpl.rcParams['figure.figsize'] = (18, 10)
    mpl.rcParams['axes.grid'] = False
    sns.set_style("whitegrid")

    # Load configuration parameters from the YAML file
    data_path = config['data']['data_path']
    out_path = config['data']['output_path']

    # Multivariate data parameters
    mwin = config['serialization']['mD']['win']
    mdeep = config['serialization']['mD']['deep']
    m_ftrs = config['serialization']['mD']['n_ftrs']

    tr_tst = config['serialization']['tr_tst']

    #Univariate data parameters
    win = config['serialization']['1D']['win']
    deep = config['serialization']['1D']['deep']
    n_ftrs = config['serialization']['1D']['n_ftrs']

    # Create the DataProcessor object
    data_processor = DataProcessor(data_path, out_path)
    data_processor.load_data()

    # Create the DataManipulation object
    data_manipulation = DataManipulation()


# Univariate data processing
    
    data_processor.process_univariate_data(win, tr_tst)
            
    fdat1 = os.path.join(out_path, "{:02}/input-output.pkl".format(win))
    lpar  = [win, deep, n_ftrs, tr_tst]

    # check if the directory exists and create it if it doesn't
    if os.path.exists(fdat1):
        data_processor.save_data(fdat1, lpar, data_processor.stock_list, data_processor.tot_res)
    else:
        directory = os.path.dirname(fdat1)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

            data_processor.save_data(fdat1, lpar, data_processor.stock_list, data_processor.tot_res)                
            print(f"File {fdat1} created and data saved.")
                                                    

# Multivariate data processing
        
    data_processor.process_multivariate_data(mwin, m_ftrs, tr_tst)

    fdat2 = os.path.join(out_path, "{:02}/m-input-output.pkl".format(mwin))
    lpar = [mwin, mdeep, m_ftrs, tr_tst]

    # check if the directory exists and create it if it doesn't
    if os.path.dirname(fdat2): 
        data_processor.save_data(fdat2, lpar, data_processor.stock_list, data_processor.tot_res)
    else:
        directory = os.path.dirname(fdat2)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

            data_processor.save_data(fdat2, lpar, data_processor.stock_list, data_processor.tot_res)
            print(f"File {fdat2} created and data saved.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)
