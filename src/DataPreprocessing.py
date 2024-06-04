'''@package DataPreprocessing
This module contains the Stock class that processes the data and creates the output files for each stock.
'''
import os
import warnings
import pandas as pd
import numpy as np
import os
import argparse
from numpy.lib.stride_tricks import sliding_window_view
from config.config import get_configuration
from utils_vv_tfg import save_data
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Stock:
    """
    Class that creates the Stock, processes the stock data, and creates the output files for each stock.
    """
    
    def __init__(self, ticker: str, file_name: str, lahead: list, tr_tst: float, scen_name: str):
        """
        Constructs all the necessary attributes for the Stock object.
        
        Parameters:
        -----------
        ticker : str
            The stock ticker.
        file_name : str
            Path to the data file.
        lahead : list
            List of the number of days ahead.
        tr_tst : float
            Train-test ratio.
        scen_name : str
            Scenario name.
        """
        self.ticker = ticker
        self._data = pd.DataFrame(self._read_data(file_name))
        self._lahead = lahead
        self.tr_tst = tr_tst
        self.df = pd.DataFrame(self._read_data(file_name))
        self.scen_name = scen_name
        self.serial_dict = {}
        self.mserial_dict = {}
        self.serial_dict['INPUT_DATA'] = {}
        self.mserial_dict['INPUT_DATA'] = {}

    def _read_data(self, file_name: str) -> pd.DataFrame:
        """
        Returns the data from the data path.
        
        Parameters:
        -----------
        file_name : str
            Path to the data.
        
        Returns:
        --------
        pd.DataFrame
            The stock data.
        """
        return pd.read_csv(file_name, sep=",", index_col=0, parse_dates=True)

    def process_univariate_data(self, win: int) -> None:
        """
        Processes the univariate data.
        
        Parameters:
        -----------
        win : int
            Window size.
        """
        for ahead in self._lahead:
            df = self.df

            win_x = sliding_window_view(df['PX_LAST'].to_numpy(), window_shape=win)[::1]
            Y = df.iloc[ahead + win:]['PX_LAST']
            X = pd.DataFrame.from_records(win_x)

            X = X.iloc[:-(ahead + 1), :]
            X.set_index(df.index[(win - 1):-(ahead + 1)], drop=True, inplace=True)

            x_mean = self.calculate_mean(X, axis=1)
            center_x = X.sub(x_mean, axis=0)
            
            min_x = min(self.calculate_min(center_x))
            max_x = max(self.calculate_max(center_x))
            vdd = pd.DataFrame({'mean': x_mean, 'min': min_x, 'max': max_x})
            vdd.set_index(X.index)

            center_x_norm = center_x.apply(lambda x: (x - min_x) / (max_x - min_x), axis=1)
            center_y_norm = pd.Series([((i - j) - min_x) / (max_x - min_x)
                                        for i, j in zip(Y.tolist(), x_mean.tolist())], index=Y.index)
            center_x_norm = center_x_norm.astype('float32')
            center_y_norm = center_y_norm.astype('float32')

            pmod = int(center_x_norm.shape[0] * self.tr_tst)
            train_x = center_x_norm.iloc[:pmod, :]
            train_y = center_y_norm.iloc[:pmod]
            test_x = center_x_norm.iloc[pmod:, :]
            test_y = center_y_norm.iloc[pmod:]
            self.serial_dict['INPUT_DATA'][ahead] = {
                                        "x": X, "y": Y, "nx": center_x_norm,
                                        "ny": center_y_norm, "numt": pmod,
                                        "trainX": train_x, "trainY": train_y,
                                        "testX": test_x, "testY": test_y, "vdd": vdd
                                        }

    def process_multivariate_data(self, mwin: int, m_ftrs: int) -> None:
        """
        Processes the multivariate data.
        
        Parameters:
        -----------
        mwin : int
            Multivariate window size.
        m_ftrs : int
            Multivariate number of features.
        """
        self.check_infinite_values(self.df)

        df_y = self.df["PX_LAST"].copy()
        df_x = self.df[["PX_LAST", "PX_OPEN", "RSI_14D", "PX_TREND", 
                        "PX_VTREND", "TWEET_POSTIVIE", "TWEET_NEGATIVE",
                        "NEWS_POSITIVE", "NEWS_NEGATIVE", "VOLATILITY"]].copy()
        idx = df_x.index[(mwin - 1):]

        multi_x, cols = self.prepare_multivariate_data(df_x, mwin, m_ftrs, idx)

        for ahead in self._lahead:
            multi_x_list = multi_x[:-(ahead + 1), :, :]
            multi_y_list = df_y.iloc[ahead + mwin:]
            mean_multi_x = self.calculate_mean(multi_x_list, axis=1)

            mean_multi_x_c = multi_x_list - mean_multi_x[:, None, :]
            pavX = pd.DataFrame(mean_multi_x, columns=df_x.columns)
            pavX.set_index(df_x.index[(mwin - 1):-(ahead + 1)], drop=True, inplace=True)
            max_mult_x = self.calculate_max(mean_multi_x_c, axis=(0, 1))
            min_mult_x = self.calculate_min(mean_multi_x_c, axis=(0, 1))
            mvdd = {"mean": pavX, "min": min_mult_x, "max": max_mult_x}
            multi_x_norm = (mean_multi_x_c - min_mult_x[None, None, :]) / (max_mult_x[None, None, :] - min_mult_x[None, None, :] + 0.00001)
            multi_y_norm = ((multi_y_list.to_numpy() - mean_multi_x[:, 0]) - min_mult_x[0]) / (max_mult_x[0] - min_mult_x[0] + 0.00001)

            multi_x_norm = multi_x_norm.astype("float32")
            multi_y_norm = multi_y_norm.astype("float32")

            multi_train_x, multi_train_y, multi_test_x, multi_test_y, pmod = self.prepare_data_for_modeling(multi_x_norm, multi_y_norm, multi=True)

            self.check_nan_values(multi_train_x, multi_test_x, ahead)

            xdx = idx[:-(ahead + 1)]
            self.mserial_dict['INPUT_DATA'][ahead] = {"x": multi_x_list, "y": multi_y_list, "nx": multi_x_norm, "ny": multi_y_norm, "numt": pmod,
                                        "trainX": multi_train_x, "trainY": multi_train_y,
                                        "testX": multi_test_x, "testY": multi_test_y, "vdd": mvdd, "cnms": cols,
                                        "idtest": xdx[pmod:]}

    def prepare_multivariate_data(self, df_x: pd.DataFrame, win: int, n_ftrs: int, idx: pd.Index) -> tuple:
        """
        Prepares the multivariate data for modeling.
        
        Parameters:
        -----------
        df_x : pd.DataFrame
            The features.
        win : int
            The window size.
        n_ftrs : int
            The number of features.
        idx : pd.Index
            The index.
        
        Returns:
        --------
        Tuple[np.ndarray, pd.Index]
            The multivariate data and columns.
        """
        multi_x = []
        cols = df_x.columns
        for i in range(len(df_x.columns)):
            lX = np.lib.stride_tricks.sliding_window_view(df_x.iloc[:, i].to_numpy(), window_shape=win)[::n_ftrs]
            ss = pd.DataFrame.from_records(lX)
            ss.set_index(idx, drop=True, inplace=True)
            multi_x.append(ss.to_numpy())
        multi_x = np.transpose(np.stack(multi_x, axis=1), (0, 2, 1))
        return multi_x, cols

    def compute_sentiment_scores(self) -> None:
        """
        Computes the sentiment scores for the data.
        """
        self.df['TWEET_POSTIVIE'] = self.df['TWITTER_POS_SENTIMENT_COUNT'] / self.df['TWITTER_PUBLICATION_COUNT'].replace(0, 1)
        self.df['TWEET_NEGATIVE'] = self.df['TWITTER_NEG_SENTIMENT_COUNT'] / self.df['TWITTER_PUBLICATION_COUNT'].replace(0, 1)
        self.df['NEWS_POSITIVE'] = self.df['NEWS_POS_SENTIMENT_COUNT'] / self.df['NEWS_PUBLICATION_COUNT'].replace(0, 1)
        self.df['NEWS_NEGATIVE'] = self.df['NEWS_NEG_SENTIMENT_COUNT'] / self.df['NEWS_PUBLICATION_COUNT'].replace(0, 1)

    def compute_volatility(self) -> None:
        '''
        Computes the daily volatility for each row in the data
        '''
        self.df['VOLATILITY'] = self.df['PX_LAST'].pct_change().rolling(window=252).std()

    def compute_trend(self) -> None:
        '''
        Computes the trend for the price and volume data
        '''
        self.df['PX_TREND'] = 2 * (self.df['PX_LAST'] - self.df['PX_OPEN']) / (self.df['PX_OPEN'] + self.df['PX_LAST'])
        self.df['PX_VTREND'] = self.df['PX_TREND'] * self.df['VOLUME']

    def check_nan_values(self, train_x: pd.DataFrame, test_x: pd.DataFrame, ahead: int) -> None:
        '''
        Checks for NaN values in the data
 
        Parameters:
        train_x : DataFrame
            the training data
        test_x : DataFrame
            the testing data
        ahead : int
            the number of days ahead
            
        Returns:
        Error message if NaN values are found'''
        if np.isinf(train_x).any() or np.isnan(train_x).any():
            print(f"Stock {self.ticker} ahead: {ahead} has nans.")
            print(train_x)

        if np.isnan(test_x).any():
            print(f"Stock {self.ticker} ahead: {ahead} has nans.")
            print(test_x)

    def check_infinite_values(self, df: pd.DataFrame) -> None:
        '''
        Checks for infinite values in the data
 
        Parameters:
        df : the DataFrame
        
        Returns:
        Error message if infinite values are found'''
        if np.isinf(df).values.sum() > 0:
            print("Error: Infinite values were found in the DataFrame.")
 
    def prepare_data_for_modeling(self, x_norm, y_norm, multi: bool) -> tuple:
        '''
        Returns the data for modeling
 
        Parameters:
        x_norm : the features
        y_norm : the target
        multi : if the data is multivariate

        Returns:
        Train_x : data for training
        Train_y : target for training
        Test_x : data for testing
        Test_y : target for testing
        '''
        pmod = int(x_norm.shape[0] * self.tr_tst)
        if multi is False:  # Univariate
            train_x = x_norm.iloc[:pmod,:]
            train_y = y_norm.iloc[:pmod]
            test_x  = x_norm.iloc[pmod:,:]
            test_y  = y_norm.iloc[pmod:]
        else:  # Multivariate
            train_x = x_norm[:pmod, :, :]
            train_y = y_norm[:pmod]
            test_x = x_norm[pmod:, :, :]
            test_y = y_norm[pmod:]
        return train_x, train_y, test_x, test_y, pmod
    
    def process_stocks(self) -> None:
        '''
        Processes the stock data'''
        self.compute_sentiment_scores()
        self.compute_volatility()
        self.compute_trend()
        self.df.dropna(inplace=True)

    def calculate_mean(self, data, axis=None) -> np.ndarray:
        '''
        Returns the mean of the data
 
        Parameters:
        data : the data
        axis : the axis
        
        Returns:
        The mean of the data'''
        return np.mean(data, axis=axis)

    def calculate_min(self, data, axis=None) -> np.ndarray:
        '''
        Returns the minimum of the data
 
        Parameters:
        data : the data
        axis : the axis
        
        Returns:
        The minimum of the data'''
        return data.min(axis=axis)

    def calculate_max(self, data, axis=None) -> np.ndarray:
        '''
        Returns the maximum of the data
 
        Parameters:
        data : the data
        axis : the axis
        
        Returns:
        The maximum of the data'''
        return data.max(axis=axis)


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


def main(args) -> None:
    '''
    Main function that processes the data and creates the output files for each stock'''
    config, _ = get_configuration(args.params_file)

    if args.verbose > 0:
        print("Additional Info:")
        # Add additional information here
        print("Processing data with the configuration file:", args.params_file)

    data_path = config['data']['data_path']
    out_path = config['data']['output_path']
    filename_structure = config['data']['filename_structure']
    date = config['data']['date']

    for scenario in config['scenarios']:
        list_win_size = scenario['win']
        lahead = scenario['lahead']
        stock_list = scenario['tickers']
        n_ftrs = scenario['n_features']
        for win in list_win_size:
            for ticker in stock_list:
                filename = filename_structure.format(ticker=ticker, date=date)
                file = os.path.join(data_path, filename)
                assert os.path.exists(file), f"El archivo {file} no existe."
                
                for tr_tst in scenario['tr_tst']:
                    scen_name = scenario['name']
                    stock = Stock(ticker, file, lahead, tr_tst, scen_name)
                    stock.process_stocks()

                    lpar = [win, n_ftrs, tr_tst]
                    # Univariate data processing
                    stock.process_univariate_data(win)
                    fdat1 = out_path + "/input/{}/{}/{}-{}-input.pkl".format(win,stock.tr_tst,stock.scen_name,ticker)

                    # Multivariate data processing
                    stock.process_multivariate_data(win, n_ftrs)
                    fdat2 = out_path + "/input/{}/{}/{}-{}-m-input.pkl".format(win,stock.tr_tst,stock.scen_name,ticker)

                    # Save univariate data
                    if not os.path.exists(fdat1):
                        directory1 = os.path.dirname(fdat1)
                        if not os.path.exists(directory1):
                            os.makedirs(directory1)
                            print(f"Directory {directory1} created.")
                            
                    save_data(fdat1, out_path, lahead, lpar, stock.serial_dict)
                    print(f"File {fdat1} data saved.")

                    # Save multivariate data
                    if not os.path.exists(fdat2):
                        directory1 = os.path.dirname(fdat2)
                        if not os.path.exists(directory1):
                            os.makedirs(directory1)
                            print(f"Directory {directory1} created.")

                    save_data(fdat2, out_path, lahead, lpar, stock.mserial_dict)
                    print(f"File {fdat2} data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")
    parser.add_argument("-c", "--params_file", nargs='?', action='store', help="Configuration file path")
    args = parser.parse_args()
    main(args)