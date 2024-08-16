'''@package OptimizePortfolio
This module looks to optimize the assets to be included in the portfolio.
'''
import os, warnings, datetime, pickle, argparse, statistics
import pandas as pd
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from config.config import get_configuration
from scipy.optimize import minimize
from utils_vv_tfg import *
from config.config import get_configuration
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Stock:
    """
    Class that creates the Stock, processes the stock data, and creates the output files for each stock.
    """
    
    def __init__(self, ticker: str, file_name: str, lahead: list, tr_tst: float, scen_name: str, sep: str = ',', encoding: str = 'utf-8'):
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
        self.sep    = sep
        self.encoding=encoding
        self._data = pd.DataFrame(self._read_data(file_name))
        self._lahead = lahead
        self.tr_tst = tr_tst
        self.df = pd.DataFrame(self._read_data(file_name))
        self.scen_name = scen_name
        self.serial_dict = {}
        self.mserial_dict = {}
        self.serial_dict['INPUT_DATA'] = {}
        self.mserial_dict['INPUT_DATA'] = {}

    def lst_data(self,nlin: int):
        pd.set_option('display.max_columns', None)
        print(self._data.head(nlin))

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
        return pd.read_csv(file_name, sep=self.sep, index_col=0, parse_dates=True, encoding=self.encoding).dropna()

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

    def calculate_min(self, data, axis=1) -> np.ndarray:
        '''
        Returns the minimum of the data
 
        Parameters:
        data : the data
        axis : the axis
        
        Returns:
        The minimum of the data'''
        return data.min(axis=axis)

    def calculate_max(self, data, axis=1) -> np.ndarray:
        '''
        Returns the maximum of the data
 
        Parameters:
        data : the data
        axis : the axis
        
        Returns:
        The maximum of the data'''
        return data.max(axis=axis)




class Portfolio:
    """
    Class that creates the Portfolio for different stocks.
    """

    def __init__(self, tickers: list, scen_name: str, cv_file: str, mat_cat: dict, \
                 lstlmb: float,t_ini: str,t_end: str,lst_treb: list):
        """
        Constructs all the necessary attributes for the Stock object.
        
        Parameters:
        -----------
        tickers : list
            The stock ticker names.
        scen_name : str
            Name of scenario under optimization
        cv_file
            Path to the data file for correlations between stocks in the scenario.
        mat_cat : dict
            Matrix of categories, having the name of stocks on each category.
        """
        self.tickers   = tickers
        self.scen_name = scen_name
        if os.path.exists(cv_file):
            with open(cv_file, "rb") as openfile:
                self.cv_path = pickle.load(openfile)
                self.cv_file = pickle.load(openfile)
                dctscen      = pickle.load(openfile)
                self._cvmat  = dctscen[self.scen_name]
        else:
            raise FileNotFoundError(f"El archivo {cv_file} no existe.")
        self.nctg   = len(mat_cat)
        self.ntkrs  = len(tickers)
        self.mat_cat= pd.DataFrame(0, index=list(mat_cat.keys()), columns=tickers)
        for ic in mat_cat.keys():
            for jc in mat_cat[ic]:
                self.mat_cat.loc[ic,jc] = 1    
                for kc in mat_cat[ic]:   # Deactivating the stocks in the same category
                    self._cvmat.loc[jc,kc] = 1
                    self._cvmat.loc[kc,jc] = 1
        self.lmbda  = lstlmb
        self.t_strt = t_ini
        self.t_end  = t_end
        self.t_reb  = lst_treb
        self.price  = {}

    def take_results(self,config_file: str)->None:
        self.config, _   = get_configuration(config_file)
        output_path      = self.config['data']['output_path']
        self.all_results = {}
        for scen in self.config['scenarios']:
            if self.scen_name == scen['name']:
                self.lst_tr_tst  = scen['tr_tst']
                self.lst_win_sz  = scen['win']
                self.lahead      = scen['lahead']
                for win in self.lst_win_sz:
                    self.all_results[win] = {}
                    self.price[win]       = {}
                    for tr_tst in self.lst_tr_tst:
                        self.all_results[win][tr_tst] = load_output_preprocessed_data( \
                                        output_path, win, tr_tst, self.scen_name)
                        l_mdls = list(self.all_results[win][tr_tst].keys())
                        self.calc_perf(l_mdls, win, tr_tst)
                self.scenario = scen

    # self.all_results[0.85]['lstm']['tot_res']['OUT_MODEL']['AAPL'][30].loc[
    #                     0]['DY'].loc[date,[['Y_real','Y_predicted']]
    def calc_perf(self, mdls:list, win: int, tr_tst: float)->None:
        lst_pt = {}
        for iml in mdls:
            if iml in self.all_results[win][tr_tst].keys():
                lst_pt[iml] = self.extrct_price(win, tr_tst, iml)
        matres = xr.Dataset(lst_pt).to_array().mean(axis=0).to_pandas()
        self.price[win][tr_tst] = self.calc_price(win, tr_tst, matres) 

    def ldates(self, win: int, tr_tst: float, mdl: str, istk: str)->dict:
        lhead = list(self.all_results[win][tr_tst][mdl]['tot_res']['OUT_MODEL'][istk].keys())
        torg  = datetime.datetime.strptime(self.t_strt,"%Y-%m-%d")
        lst_dts = {}
        ts= self.all_results[win][tr_tst][mdl]['tot_res']['OUT_MODEL' \
                                    ][istk][lhead[0]].loc[0,'DY'].index.tolist()
        # Find the nearest origin available.
        lst_dts[0] = self.find_near_date(torg,0,ts)
        for ahd in lhead:
            ts= self.all_results[win][tr_tst][mdl]['tot_res']['OUT_MODEL' \
                                    ][istk][ahd].loc[0,'DY'].index.tolist()
            lst_dts[ahd] = self.find_near_date(torg,ahd,ts)
        return(lst_dts)

    def find_near_date(self, torg : datetime, ahd: int, ts: list) -> str :
        tmp_org = torg + datetime.timedelta(days=ahd)
        if tmp_org < min(ts) or tmp_org > max(ts):
            return(None) 
        while tmp_org not in ts:
            tmp_org = tmp_org + datetime.timedelta(days=1)
        return(tmp_org.strftime("%Y-%m-%d"))

    def extrct_price(self, win: int, tr_tst: float, mdl: str) -> pd.DataFrame :
        pts   = pd.DataFrame(0, index=self.tickers, columns= [0]+self.lahead)
        stcks = list(self.all_results[win][tr_tst][mdl]['tot_res']['OUT_MODEL'].keys())
        self.dts_stk = {}
        for istk in self.tickers:
            lhead = list(self.all_results[win][tr_tst][mdl]['tot_res'][\
                                'OUT_MODEL'][istk].keys())
            self.dts_stk[istk] = self.ldates(win,tr_tst,mdl,istk)
            for id in self.dts_stk[istk].keys():
                idt = datetime.datetime.strptime(self.dts_stk[istk][id],"%Y-%m-%d")
                if id == 0: # Reference Price
                    pts.loc[istk,id] = self.all_results[win][tr_tst][mdl]['tot_res'][ \
                        'OUT_MODEL'][istk][lhead[0]].loc[id,'DY'].loc[idt,'Y_real']
                else: 
                    tmp = self.all_results[win][tr_tst][mdl][\
                            'tot_res'][ 'OUT_MODEL'][istk][id]
                    vl = []
                    for jtmp in tmp.index:
                        vl.append(tmp.loc[jtmp,'DY'].loc[idt,'Y_predicted'])
                    pts.loc[istk,id] = statistics.mean(vl)
        return(pts)
    
    def calc_price(self, win: int, tr_tst: float, matres:pd.DataFrame) -> pd.DataFrame:
        price = pd.DataFrame(0, index=self.tickers, columns= self.lahead)
        for id in price.columns:
            for istk in self.tickers:
                price.loc[istk,id] = 100.*np.log(matres.loc[istk,id]/matres.loc[istk,0])
        return(price)
    
    def get_cvmat(self):
        return(self._cvmat)
    
    def get_price(self, win: int, tr_tst:float):
        return(self.price[win][tr_tst])
    
    def get_ntks(self):
        return(self.ntkrs)
    
    def get_lambda(self):
        return(self.lmbda)

    def get_scenario(self):
        return(self.scenario)
    
    def get_trtst(self):
        return(self.lst_tr_tst)

    def get_mcats(self):
        return(self.mat_cat)

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


def objective(x: list, prms: dict) -> float:
    ptf   = prms['Portfolio'] 
    win   = prms['win']
    tr_tst= prms['tr_tst']
    dlt   = prms['TN'] # Days of portfolio life
    #
    ncs   = ptf.get_ntks()
    lmda  = ptf.get_lambda()
    cvar  = ptf.get_cvmat()
    price = ptf.get_price(win,tr_tst)
    stks  = cvar.columns
    #
    kk    = pd.Series(x).T
    kk.index = stks
    y     = cvar.dot(kk)
    val1  = kk.T.dot(y)
    val2  = 0
    for j in stks:
        idx = stks.get_loc(j)
        val2= val2 + price.loc[j,dlt]*x[idx]
    return(val1 - lmda*val2)

def evalres(x: list, prms: dict) -> list:
    ptf   = prms['Portfolio'] 
    win   = prms['win']
    tr_tst= prms['tr_tst']
    dlt   = prms['TN'] # Days of portfolio life
    #
    ncs   = ptf.get_ntks()
    lmda  = ptf.get_lambda()
    cvar  = ptf.get_cvmat()
    price = ptf.get_price(win,tr_tst)
    stks  = cvar.columns
    #
    kk    = pd.Series(x).T
    kk.index = stks
    y     = cvar.dot(kk)
    val1  = kk.T.dot(y)
    val2  = 0
    for j in stks:
        idx = stks.get_loc(j)
        val2= val2 + price.loc[j,dlt]*x[idx]
    return([val1, val2])

def Fnret(x):
    global cat_mat, minNcat, stock_list
    kk = pd.Series(x).T
    kk.index = stock_list
    y  = cat_mat.dot(kk)
    return(int(sum(z > 0.0001 for z in y)) - minNcat)

def Brhs(x):  # The sum of X must be 1
    return(np.sum(x)-1)

def main(args) -> None:
    '''
    Main function that processes the data and creates the output files for each stock'''
    global minNcat, stock_list, cat_mat
    config, _ = get_configuration(args.params_file)
    verbose = 0
    if args.verbose is not None:
        verbose = int(args.verbose)
        print("Additional Info:")
        # Add additional information here
        print("Processing data with the configuration file:", args.params_file)

    data_path= config['data']['data_path']
    out_path = config['data']['output_path']
    sep      = config['data']['sep']
    encoding = config['data']['encoding']
    filename_structure = config['data']['filename_structure']
    date     = config['data']['date']
    prtfl_sc = config['portfolio']['scenario']
    lst_lmb  = config['portfolio']['lambda']
    mat_cat  = config['portfolio']['mat_cat']
    t_ini    = config['portfolio']['tstrt']
    t_end    = config['portfolio']['tend']
    lst_treb = config['portfolio']['treb']
    minNcat  = config['portfolio']['min_cats']
    selwin   = config['portfolio']['selwin']
    x0       = config['portfolio']['x0']
    optim    = config['portfolio']['optim']
    prms     = {}
    cvmat    = out_path + "/input/" + prtfl_sc + "-corr.pkl"
    # Create the Portfolio object from the config.
    for scenario in config['scenarios']:
        if scenario['name'] == prtfl_sc:
            lst_win_sz = scenario['win']
            lahead     = scenario['lahead']
            stock_list = scenario['tickers']
            prtflio    = Portfolio(stock_list, prtfl_sc, cvmat, mat_cat, lst_lmb, \
                                   t_ini, t_end, lst_treb)
    # Extract the information from price forecast
    prtflio.take_results(args.params_file)
    cat_mat = prtflio.get_mcats()
    # 
    prms['Portfolio'] = prtflio
    prms['win']       = (prtflio.get_scenario()['win'])[selwin]
    prms['tr_tst']    = (prtflio.get_trtst())[0]
    prms['TN']        = 90 # Days of portfolio life
    #
    rest = {}
    x0   = [0] * prtflio.get_ntks()
    x0[0]= 0.5
    x0[4]= 0.5
    bnds = [(0,1)] * prtflio.get_ntks()
    cons = [{'type': 'eq', 'fun': Brhs},
            {'type': 'ineq', 'fun': Fnret}]
    if optim > 0:
        res  = minimize(objective, x0 = x0, args=(prms,), constraints=cons, \
                bounds=bnds,)
        rest['x']   = res.x
        xsol        = pd.Series(rest['x'].T)
        rest['fun'] = res.fun
        rest['nit'] = res.nit
        rest['st']  = res.status
        rest['sccs']= res.success
        rest['mCat']= minNcat
    else:
        rest['x']   = x0
        xsol        = pd.Series(x0)
        rest['fun'] = objective(x0,prms)
        rest['nit'] = 0
        rest['st']  = 0
        rest['sccs']= True
        rest['mCat']= minNcat
    print(rest)
    xsol.index = stock_list
    lvals= evalres(rest['x'],prms)
    #
    res = pd.DataFrame()
    res.loc[0,xsol.index] = xsol
    res.loc[0,'optm']     = optim
    res.loc[0,'minCat']   = rest['mCat']
    res.loc[0,'nit']      = rest['nit']
    res.loc[0,'Card(Stk)']= np.sum(x > 0.001 for x in xsol)
    res.loc[0,'lambda']   = lst_lmb
    res.loc[0,'f(x)']     = rest['fun']
    res.loc[0,'var']      = lvals[0]
    res.loc[0,'perf']     = lvals[1]
    res.loc[0,'status']   = rest['st']
    res.loc[0,'success']  = rest['sccs']
    res.loc[0,'dtime']    = datetime.datetime.strftime(datetime.datetime.now(),\
                                                       "%Y-%m-%d %H:%M:%S")
    res.style.format(precision=4)
    fout         = out_path + '/res_optim.xlsx'
    if os.path.exists(fout):
        sheet = pd.read_excel(fout)
        sheet = pd.concat([sheet,res], ignore_index=True)
    else:
        sheet = res
    sheet.to_excel(fout, sheet_name='Optim', index=None)
    #
    print(xsol.index,"\t",xsol.values)
    print(f'Lambda=,\t{lst_lmb:.5f}')
    fres = rest['fun']
    print(f'F(X)=\t{fres:.6f}')
    print(f'Variance:{lvals[0]:.6f} \t Perf/$USD:{lvals[1]:.3f}')

#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose', help="Option for detailed information")
    parser.add_argument("-c", "--params_file", nargs='?', action='store', help="Configuration file path")
    args = parser.parse_args()
    main(args)
