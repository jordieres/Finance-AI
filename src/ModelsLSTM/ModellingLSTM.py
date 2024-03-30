import os, time, gc, sys, io
import datetime, pickle, session_info
import warnings, random, math
#
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mpl_toolkits.axisartist as AA
#
from scipy import stats
from pandas import Series
from argparse import ArgumentParser
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import host_subplot
# from keras_self_attention import SeqSelfAttention
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import yaml

sys.path.append('D:\Escritorio\TFG\Finance-AI\src\DataPreprocessing')

from DataPreprocessing import DataManipulation, DataProcessor, Normalizer

# tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

processed_path = 'D:/Escritorio/TFG/Finance-AI/DataProcessed'
win_size = 22

processed_path, fdat, lahead, lpar, stock_list, tot_res, df_dict = DataManipulation.load_preprocessed_data(processed_path, win_size)

win, deep, n_ftrs,tr_tst = lpar

def eval(nptstX, nptstY, testX, model, vdd, Y, ahead):
    '''
    Returns the evaluation of the model with the test data

    Arguments:
    nptstX - normalized test data
    nptstY - normalized test labels
    model - trained model being evaluated
    vdd - validation dataframe storing the data for normalization and denormalization
    Y - output PX_OPEN stock
    '''

    Yhat   = model.predict(nptstX,verbose=0)
    tans   = Yhat.shape
    if len(tans) > 2 and tans[1] == 1:
        Yhat= np.concatenate(Yhat,axis=0)
    if len(tans) > 2 and tans[1] > 1:
        Yhat= [vk[0].tolist() for vk in Yhat]
    Yprd   = np.concatenate(Yhat,axis=0).tolist()
    mse    = mean_squared_error(Yprd , nptstY)
    eff    = pd.DataFrame({'Yorg':nptstY,'Yfct':Yprd}) 
    eff.set_index(testX.index,drop=True,inplace=True)

    jdx = testX.index
    Yf = Normalizer.denormalize_data(Yprd, vdd, jdx) # denormalize the forecast
    Yr  = Y.loc[jdx] # y real
    Yy  = Y.shift(ahead).loc[jdx] # Y yesterday
    DY  = pd.concat([Yr,Yf,Yy],axis=1)
    DY.columns = ['Y_real','Y_predicted','Y_yesterday']

    msep= mean_squared_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    msey= mean_squared_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real

    return({'msep':msep,'msey':msey,'Ys':DY, 'eff': eff})



def lstm_fun(trainX,trainY,testX,testY,Y,vdd,epoch,bsize,nhn,win,n_ftrs,ahead,stock,seed):
    '''
    LSTM model 

    Returns the evaluation of the model with the test data
    
    Arguments:
    trainX - normalized training data
    trainY - normalized training labels
    testX - normalized test data
    testY - normalized test labels
    Y - output PX_OPEN stock
    vdd - validation dataframe storing the data for normalization and denormalization
    epoch - number of epochs for training
    bsize - batch size for feeding the model
    nhn - number of hidden neurons
    win - window size (days considered to learn from == trainX.shape[1])
    n_ftrs - number of expected outputs (1 in our case)
    stock - stock being evaluated
    ahead - shift value for the stock
    seed - seed to stabilize the repetitions
    '''

    nptrX = trainX.to_numpy().reshape(trainX.shape[0],trainX.shape[1],n_ftrs)
    nptrY = trainY.to_numpy()
    nit  = 0
    lloss= np.nan
    while math.isnan(lloss) and nit < 5:
        tf.random.set_seed(seed)
        # create a very basic LSTM model
        model = Sequential()
        model.add(LSTM(nhn, activation='relu', input_shape=(win,n_ftrs))) 
        model.add(Dense(n_ftrs)) # Output of a single value
        model.compile(loss='mean_squared_error', optimizer='adam')
        #
        hist = model.fit(nptrX, nptrY, epochs=epoch, batch_size=bsize, verbose=0)
        lloss= hist.history['loss'][-1]
        nit  = nit + 1
    # Predict
    nptstX = testX.to_numpy().reshape(testX.shape[0],testX.shape[1],n_ftrs)
    nptstY = testY.to_numpy()
    res1   = eval(nptstX, nptstY, testX, model, vdd, Y, ahead)
    df_result = {'MSEP':res1.get("msep"),'MSEY': res1.get("msey"),'Stock':stock,
                 'DY':res1.get("Ys"),'ALG':'LSTM','seed':seed,'epochs':epoch,
                 'nhn':nhn,'win':win,'ndims':1, 'lossh':lloss, 'nit':nit,
                 'model':model}
    return(df_result)


def stck_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epoch,bsize,nhn,win,n_ftrs,ahead,stock,seed):
    '''
    Stack LSTM model

    Returns the evaluation of the model with the test data
    
    Arguments:
    trainX - normalized training data
    trainY - normalized training labels
    testX - normalized test data
    testY - normalized test labels
    Y - output PX_OPEN stock
    vdd - validation dataframe storing the data for normalization and denormalization
    epoch - number of epochs for training
    bsize - batch size for feeding the model
    nhn - number of hidden neurons
    win - window size (days considered to learn from == trainX.shape[1])
    n_ftrs - number of expected outputs (1 in our case)
    stock - stock being evaluated
    ahead - shift value for the stock
    seed - seed to stabilize the repetitions
    '''
    nptrX = trainX.to_numpy().reshape(trainX.shape[0],trainX.shape[1],n_ftrs)
    nptrY = trainY.to_numpy()
    nit  = 0
    lloss= np.nan
    while math.isnan(lloss) and nit < 5:
        tf.random.set_seed(seed)
        # create a very Stcked-LSTM model
        stmodel = Sequential()
        stmodel.add(LSTM(nhn, activation='relu', return_sequences=True, input_shape=(win,n_ftrs)))
        stmodel.add(LSTM(nhn, activation='relu', return_sequences=True))
        stmodel.add(LSTM(nhn, activation='relu'))
        stmodel.add(Dense(n_ftrs)) # Output of a single value
        stmodel.compile(loss='mean_squared_error', optimizer='adam')
        hist = stmodel.fit(nptrX, nptrY, epochs=epoch, batch_size=bsize, verbose=0)
        lloss= hist.history['loss'][-1]
        nit  = nit + 1
    # Predict
    nptstX = testX.to_numpy().reshape(testX.shape[0],testX.shape[1],n_ftrs)
    nptstY = testY.to_numpy()
    res1   = eval(nptstX, nptstY, testX, stmodel, vdd, Y, ahead)
    df_result = {'MSEP':res1.get("msep"),'MSEY': res1.get("msey"),'Stock':stock,
                 'DY':res1.get("Ys"),'ALG':'STACK-LSTM','seed':seed,'epochs':epoch,
                 'nhn':nhn,'win':win ,'ndims':1, 'lossh':lloss, 'nit':nit,
                 'model':stmodel}
    return(df_result)


def main(args):

    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    f.close()

    epochs= config['LSTM']['epochs']
    bsize= config['LSTM']['batch_size']
    nhn  = config['LSTM']['nhn']
    res  = {}
    tmod =  config['LSTM']['model']    # lstm stcklstm, cnnlstm or attlstm
    res['MODEL'] = tmod
    fmdls = 'D:/Escritorio/TFG/Finance-AI/Models/{}'.format(nhn)+tmod+'/'
    if not os.path.exists(fmdls):
        os.makedirs(fmdls)
    #
    # for stock in stock_list:
        # res[stock] = {}
    stock = 'AAPL'
    res[stock] = {}
    for ahead in lahead:
        # print('Training ' + stock)
        trainX = tot_res['INP'][stock][ahead]['trX']
        trainY = tot_res['INP'][stock][ahead]['trY']
        testX  = tot_res['INP'][stock][ahead]['tsX']    
        testY  = tot_res['INP'][stock][ahead]['tsY']
        Y      = tot_res['INP'][stock][ahead]['y']
        vdd    = tot_res['INP'][stock][ahead]['vdd']
        tmpr = []
        
        for irp in range(15):
            seed      = random.randint(0,1000)
            lstm_start= time.time()
            mdl_name  = '{}-{}-{:03}-{:02}.hd5'.format(tmod,stock,ahead,irp)
            if tmod == "lstm":
                sol   = lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win_size,n_ftrs,ahead,stock,seed)
            if tmod == "stcklstm":
                sol   = stck_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win_size,n_ftrs,ahead,stock,seed)
            '''if tmod == "cnnlstm":
                sol   = cnn_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win_size,n_ftrs,ahead,stock,seed)
            if tmod == "attlstm":
                sol   = att_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win_size,n_ftrs,ahead,stock,seed)'''
            lstm_end  = time.time()
            ttrain    = lstm_end - lstm_start
            sol['ttrain'] = ttrain
            sol['epochs']  = epochs
            sol['bsize']  = bsize
            sol['nhn']    = nhn
            sol['model'].save(fmdls+mdl_name)
            sol['model']  = fmdls+mdl_name
            print('   Effort spent: ' + str(ttrain) +' s.')
            sys.stdout.flush()
            tmpr.append(sol)
        res[stock][ahead] = pd.DataFrame(tmpr)
    
    tot_res['OUT_MODEL'] = res

    data_path = config['data']['data_path']
    
    fdat2 = 'D:/Escritorio/TFG/Finance-AI/DataProcessed/model-'+tmod+'-output.pkl'

    file = open(fdat2, 'wb')
    pickle.dump(data_path, file)
    pickle.dump(fdat2,file)
    pickle.dump(lahead,file)
    pickle.dump(lpar, file)
    pickle.dump(stock_list,file)
    pickle.dump(tot_res, file)
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)
