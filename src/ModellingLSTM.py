import os, sys
import time
import random
import math
import argparse
import warnings
import json
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_self_attention import SeqSelfAttention

from utils_vv_tfg import save_data, load_preprocessed_data, eval_lstm
from config.config import get_configuration

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class TrainLSTM:
    def __init__(self, trainX: pd.DataFrame, trainY: pd.DataFrame, 
                 testX: pd.DataFrame, testY: pd.DataFrame, Y: pd.DataFrame,
                 vdd: dict, epoch: int, bsize: int, nhn: int, win: int,
                 n_ftrs: int, stock: str):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.Y = Y
        self.vdd = vdd
        self.epoch = epoch
        self.bsize = bsize
        self.nhn = nhn
        self.win = win
        self.n_ftrs = n_ftrs
        self.stock = stock

    def lstm_fun(self, ahead: int, seed: int) -> dict:
        """Trains a basic LSTM model and evaluates its performance.

        Args:
            ahead (int): Number of days ahead for forecasting.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the LSTM model training and evaluation.
        """
        nptrX = self.trainX.to_numpy().reshape(
            self.trainX.shape[0],self.trainX.shape[1],self.n_ftrs)
        nptrY = self.trainY.to_numpy()
        nit  = 0
        lloss= np.nan
        while math.isnan(lloss) and nit < 5:
            tf.random.set_seed(seed)
            model = Sequential()
            model.add(LSTM(self.nhn, activation='relu', input_shape=(self.win, self.n_ftrs)))
            model.add(Dense(self.n_ftrs))  # Output of a single value
            model.compile(loss='mean_squared_error', optimizer='adam')
            checkpoint = ModelCheckpoint('model-basic.keras', verbose=1, \
                    monitor='val_loss',save_best_only=True, mode='auto') 
            hist = model.fit(nptrX, nptrY, epochs=self.epoch, batch_size=self.bsize, verbose=0, callbacks=[checkpoint])
            lloss= hist.history['loss'][-1]
            nit  = nit + 1
        # Predict
        nptstX = self.testX.to_numpy().reshape(
            self.testX.shape[0],self.testX.shape[1],self.n_ftrs)
        nptstY = self.testY.to_numpy()
        res1   = eval_lstm(nptstX, nptstY, self.testX, 
                           model, self.vdd, self.Y, ahead)
        df_result = {'MSEP':res1.get("msep"),'MSEY': res1.get("msey"),
                     'MAEP':res1.get("maep"),'MAEY': res1.get("maey"),
                     'Stock':self.stock, 'DY':res1.get("Ys"),'nhn':self.nhn,
                     'win':self.win,'ndims':1, 'lossh':lloss, 'nit':nit,
                    'model':model}
        return(df_result)


    def stck_lstm_fun(self, ahead: int, seed: int) -> dict:
        """Trains a stacked LSTM model and evaluates its performance.

        Args:
            ahead (int): Number of days ahead for forecasting.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the stacked LSTM model training and evaluation.
        """
        nptrX = self.trainX.to_numpy().reshape(
            self.trainX.shape[0],self.trainX.shape[1],self.n_ftrs)
        nptrY = self.trainY.to_numpy()
        nit  = 0
        lloss= np.nan
        while math.isnan(lloss) and nit < 5:
            tf.random.set_seed(seed)
            stmodel = Sequential()
            stmodel.add(LSTM(self.nhn, activation='relu', return_sequences=True,
                              input_shape=(self.win,self.n_ftrs)))
            stmodel.add(LSTM(self.nhn, activation='relu', return_sequences=True))
            stmodel.add(LSTM(self.nhn, activation='relu'))
            stmodel.add(Dense(self.n_ftrs)) # Output of a single value
            stmodel.compile(loss='mean_squared_error', optimizer='adam')
            checkpoint = ModelCheckpoint('model-stacked.keras', verbose=1, \
                    monitor='val_loss',save_best_only=True, mode='auto') 
            hist = stmodel.fit(nptrX, nptrY, epochs=self.epoch, batch_size=self.bsize, verbose=0, callbacks=[checkpoint])
            lloss= hist.history['loss'][-1]
            nit  = nit + 1
        # Predict
        nptstX = self.testX.to_numpy().reshape(
            self.testX.shape[0],self.testX.shape[1],self.n_ftrs)
        nptstY = self.testY.to_numpy()
        res1   = eval_lstm(nptstX, nptstY, self.testX, stmodel, self.vdd, self.Y, ahead)
        df_result = {'MSEP':res1.get("msep"),'MSEY': res1.get("msey"),
                     'MAEP':res1.get("maep"),'MAEY': res1.get("maey"),
                     'Stock':self.stock,'DY':res1.get("Ys"),
                     'ALG':'STACK-LSTM','seed':seed,'epochs':self.epoch,
                     'nhn':self.nhn,'win':self.win ,'ndims':1,
                     'lossh':lloss, 'nit':nit,'model':stmodel}
        return(df_result)

    def att_lstm_fun(self, ahead: int, seed: int) -> dict:
        """Trains an attention-based LSTM model and evaluates its performance.

        Args:
            ahead (int): Number of days ahead for forecasting.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the attention-based LSTM model training and evaluation.
        """
        aptrX = self.trainX.to_numpy().reshape(
            self.trainX.shape[0], self.win,self.n_ftrs)
        aptrY = self.trainY.to_numpy()
        nit  = 0
        lloss= np.nan
        while math.isnan(lloss) and nit < 5:
            amodel = Sequential()
            amodel.add(Embedding(input_dim=self.win,output_dim=int(1),
                            mask_zero=True))
            amodel.add(GRU(self.nhn, activation='relu', \
                    return_sequences=True,input_shape=(self.win,self.n_ftrs)))
            amodel.add(SeqSelfAttention(attention_activation='sigmoid'))
            amodel.add(Dense(self.n_ftrs))
            amodel.compile(loss='mean_squared_error', optimizer='adam')
            earlyStopping = EarlyStopping(monitor='val_loss', \
                                patience=10, verbose=0, mode='min')
            checkpoint = ModelCheckpoint('model-attention.h5', verbose=1, \
                    monitor='val_loss',save_best_only=True, mode='auto')  
            hist = amodel.fit(aptrX, aptrY, epochs=self.epoch, callbacks=[checkpoint], \
                        validation_split=0.15,batch_size=self.bsize,verbose=1)
            lloss= hist.history['val_loss'][-1]
            nit  = nit + 1
            # Predict
        amodel = load_model('model-attention.h5',custom_objects={
                'SeqSelfAttention': SeqSelfAttention})
        aptstX = self.testX.to_numpy().reshape(
            self.testX.shape[0],self.win,self.n_ftrs)
        aptstY = self.testY.to_numpy()
        res5b   = eval_lstm(aptstX, aptstY, self.testX, amodel, self.vdd, self.Y, ahead)
        #
        df_result = {'MSEP': res5b.get("msep"),'MSEY': res5b.get("msey"),
                    'MAEP':res5b.get("maep"),'MAEY': res5b.get("maey"),
                    'Stock': self.stock,'DY':res5b.get("Ys"),
                    'ALG':'ATT_LSTM',
                    'seed':seed,'epochs':self.epoch,'nhn':self.nhn,
                    'win':self.win, 'ndims':1,'lossh':lloss, 'nit':nit,
                    'model':amodel}
        return(df_result)

def main(args) -> None:
    """Main function to train LSTM models for time series forecasting.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    config, _ = get_configuration(args.params_file)
    processed_path = config['data']['output_path']
    multi = False
    
    lstm_configs = []
    scenarios = []
    for scenario in config['scenarios']:
        list_win_size = scenario['win']
        list_tr_tst = scenario['tr_tst']
        lahead = scenario['lahead']
        stock_list = scenario['tickers']
        epochs = scenario['epochs']
        n_itr = scenario['n_itr']
        bsize = scenario['batch_size']
        nhn = scenario['nhn']
        if 'LSTM' in scenario:
            lstm_configs.append(scenario['LSTM'])
            scenarios.append(scenario['name'])
        res = {}
        for i, config_lstm in enumerate(lstm_configs):
            tmod = config_lstm['model']    # lstm stcklstm or attlstm
            scen_model = 'MODEL_'+scenario['name']
            res[scen_model] = tmod
            for win_size in list_win_size:
                for tr_tst in list_tr_tst:
                    out_model = {}
                    for stock in stock_list:
                        lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, scenario['name'], multi)
                        win, n_ftrs, tr_tst = lpar

                        fmdls = config['data']['fmdls'].format(nhn=nhn,tmod=tmod,tr_tst=tr_tst,stock=stock)
                        if not os.path.exists(fmdls):
                            os.makedirs(fmdls)
                        res[stock] = {}
                        print(f"Traning for {win} window size {tr_tst*100}% training data")
                        for ahead in lahead:
                            tot = tot_res['INPUT_DATA'][ahead]
                            trainX = tot['trainX']
                            trainY = tot['trainY']
                            testX  = tot['testX']    
                            testY  = tot['testY']
                            Y      = tot['y']
                            vdd    = tot['vdd']
                            tmpr = []
                            model_lstm = TrainLSTM(trainX, trainY, testX, testY, Y, vdd, epochs, bsize, nhn, win, n_ftrs, stock)
                            for irp in range(n_itr):
                                seed      = random.randint(0,1000)
                                print('######################################################')
                                print('Training ' + stock + ' ahead ' + str(ahead) + ' days.')
                                lstm_start= time.time()
                                mdl_name  = f'{tmod}-{stock}-{ahead}-{irp}'
                                if tmod == "lstm":
                                    sol   = model_lstm.lstm_fun(ahead, seed)
                                if tmod == "stcklstm":
                                    sol   = model_lstm.stck_lstm_fun(ahead, seed)
                                if tmod == "attlstm":
                                    sol   = model_lstm.att_lstm_fun(ahead, seed)
                                lstm_end  = time.time()
                                ttrain    = lstm_end - lstm_start
                                sol['ttrain'] = ttrain
                                sol['epochs']  = epochs
                                sol['bsize']  = bsize
                                sol['nhn']    = nhn
                                sol['win']    = win
                                sol['tr_tst'] = tr_tst
                                model_json 	= {}
                                model_json['model'] = sol['model'].to_json()
                                model_json['vdd'] 	 = vdd.to_json()
                                with open(f"{fmdls}{mdl_name}.json","w") as json_file:
                                    json.dump(model_json, json_file)
                                # 	serialize weights to HDF5
                                sol['model'].save_weights(f"{fmdls}{mdl_name}.h5")
                                print("Saved model to disk")
                                #sol['model'].save(fmdls+mdl_name)
                                sol['model']  = fmdls+mdl_name
                                print('   Effort spent: ' + str(ttrain) +' s.')
                                sys.stdout.flush()
                                tmpr.append(sol)
                            res[stock][ahead] = pd.DataFrame(tmpr)
                        out_model[stock] = res[stock]
                        
                    tot_res['OUT_MODEL'] = out_model
                    fdat = '{}/output/{}/{}/{}-{}-output.pkl'.format(processed_path, win,tr_tst,scenario['name'],tmod)
                    if os.path.exists(fdat):
                        save_data(fdat, processed_path, lahead, lpar, tot_res)
                    else:
                        directory1 = os.path.dirname(fdat)
                        if not os.path.exists(directory1):
                            os.makedirs(directory1)
                            print(f"Directory {directory1} created.")

                        save_data(fdat, processed_path, lahead, lpar, tot_res)                
                        print(f"File {fdat} created and data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM training, testing and results saving.")
    parser.add_argument("-c", "--params_file", nargs='?', action='store', help="Configuration file path")
    args = parser.parse_args()
    main(args)
