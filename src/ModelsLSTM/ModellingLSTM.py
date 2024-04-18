import os, time, sys
import warnings, random, math
#
import pandas as pd
import numpy as np
import tensorflow as tf
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
from keras_self_attention import SeqSelfAttention
import argparse
import yaml

sys.path.append('D:\Escritorio\TFG\Finance-AI\src')

from utils_tfg import save_data, load_preprocessed_data, denormalize_data

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

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
        Yf = denormalize_data(Yprd, vdd, jdx) # denormalize the forecast
        Yr  = Y.loc[jdx] # y real
        Yy  = Y.shift(ahead).loc[jdx] # Y yesterday
        DY  = pd.concat([Yr,Yf,Yy],axis=1)
        DY.columns = ['Y_real','Y_predicted','Y_yesterday']

        msep= mean_squared_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
        msey= mean_squared_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real

        return({'msep':msep,'msey':msey,'Ys':DY, 'eff': eff})


class TrainLSTM:

    def lstm_fun(trainX, trainY, testX, testY, Y, vdd, epoch, bsize, nhn, win, n_ftrs, ahead, stock, seed):
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


    def stck_lstm_fun(trainX, trainY, testX, testY, Y, vdd, epoch, bsize, nhn, win, n_ftrs, ahead, stock, seed):
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

    def att_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epoch,bsize,nhn,win,n_ftrs,ahead,stock,seed):
        '''
        Returns the evaluation of the model with the test data
        
        Parameters:
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
        ahead - shift value for the stock
        stock - stock being evaluated
        seed - seed to stabilize the repetitions
        '''

        aptrX = trainX.to_numpy().reshape(trainX.shape[0], win,n_ftrs)
        aptrY = trainY.to_numpy()
        nit  = 0
        lloss= np.nan
        while math.isnan(lloss) and nit < 5:
            amodel = Sequential()
            amodel.add(Embedding(input_dim=win,output_dim=int(win//3),
                            mask_zero=True))
            amodel.add(GRU(nhn, activation='relu', \
                    return_sequences=True,input_shape=(win,n_ftrs)))
            amodel.add(SeqSelfAttention(attention_activation='sigmoid'))
            amodel.add(Dense(n_ftrs))
            amodel.compile(loss='mean_squared_error', optimizer='adam')
            earlyStopping = EarlyStopping(monitor='val_loss', \
                                patience=10, verbose=0, mode='min')
            checkpoint = ModelCheckpoint('model-attention.h5', verbose=1, \
                    monitor='val_loss',save_best_only=True, mode='auto')  
            hist = amodel.fit(aptrX, aptrY, epochs=epoch, callbacks=[checkpoint], \
                        validation_split=0.15,batch_size=bsize,verbose=1)
            lloss= hist.history['val_loss'][-1]
            nit  = nit + 1
            # Predict
        amodel = load_model('model-attention.h5',custom_objects={
                'SeqSelfAttention': SeqSelfAttention})
        aptstX = testX.to_numpy().reshape(testX.shape[0],win,n_ftrs)
        aptstY = testY.to_numpy()
        res5b   = eval(aptstX, aptstY, testX, amodel, vdd, Y, ahead)
        #
        df_result = {'MSEP': res5b.get("msep"),
                    'MSEY': res5b.get("msey"), 'Stock': stock,
                    'DY':res5b.get("Ys"), 'ALG':'ATT_LSTM',
                    'seed':seed,'epochs':epoch,'nhn':nhn,
                    'win':win, 'ndims':1,'lossh':lloss, 'nit':nit,
                    'model':amodel}
        return(df_result)

def main(args):

    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    f.close()

    
    processed_path = config['data']['output_path']
    win_size = config['LSTM']['window']
    epochs = config['LSTM']['epochs']
    bsize = config['LSTM']['batch_size']
    nhn = config['LSTM']['nhn']
    res = {}
    tmod = config['LSTM']['model']    # lstm stcklstm or attlstm
    res['MODEL'] = tmod
    stock = args.stock
    multi = config['multi']
    list_tr_tst = config['tr_tst']

    for tr_tst in list_tr_tst:
        _, _, lahead, lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, multi)
        win, n_ftrs, tr_tst, deep = lpar

        fmdls = f'D:/Escritorio/TFG/Finance-AI/Models/{nhn}{tmod}/{tr_tst}/{stock}/'
        if not os.path.exists(fmdls):
            os.makedirs(fmdls)
        
        trainlstm = TrainLSTM
        res[stock] = {}
        for ahead in lahead:
            print('Training ' + stock + ' ahead ' + str(ahead) + ' days.')
            trainX = tot_res[ahead]['trainX']
            trainY = tot_res[ahead]['trainY']
            testX  = tot_res[ahead]['testX']    
            testY  = tot_res[ahead]['testY']
            Y      = tot_res[ahead]['y']
            vdd    = tot_res[ahead]['vdd']
            tmpr = []
            
            for irp in range(10):
                seed      = random.randint(0,1000)
                lstm_start= time.time()
                mdl_name  = '{}-{}-{}-{:02}.hd5'.format(tmod,stock,ahead,irp)
                if tmod == "lstm":
                    sol   = TrainLSTM.lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,seed)
                if tmod == "stcklstm":
                    sol   = trainlstm.stck_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,seed)
                if tmod == "attlstm":
                    sol   = trainlstm.att_lstm_fun(trainX,trainY,testX,testY,Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,seed)
                lstm_end  = time.time()
                ttrain    = lstm_end - lstm_start
                sol['ttrain'] = ttrain
                sol['epochs']  = epochs
                sol['bsize']  = bsize
                sol['nhn']    = nhn
                sol['win']    = win
                sol['tr_tst'] = tr_tst
                sol['model'].save(fmdls+mdl_name)
                sol['model']  = fmdls+mdl_name
                print('   Effort spent: ' + str(ttrain) +' s.')
                sys.stdout.flush()
                tmpr.append(sol)
            res[ahead] = pd.DataFrame(tmpr)
        
        tot_res['OUT_MODEL'] = res

        data_path = config['data']['data_path']
        processed_path = config['data']['output_path']
        
        fdat = f'D:/Escritorio/TFG/Finance-AI/DataProcessed/output/{win}/{tr_tst}/{stock}-{tmod}-output.pkl'
        if os.path.exists(fdat):
            save_data(fdat, processed_path, lahead, lpar, tot_res)
        else:
            directory1 = os.path.dirname(fdat)
            if not os.path.exists(directory1):
                os.makedirs(directory1)
                print(f"Directory {directory1} created.")

            save_data(fdat, processed_path, lahead, lpar, tot_res)                
            print(f"File {fdat} created and data saved.")

    # save_data(fdat, processed_path, lahead, lpar, tot_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and create output.")
    parser.add_argument("params_file", help="Path to the configuration YAML file.")
    parser.add_argument("--stock", help="Ticker of the stock to be used.")
    args = parser.parse_args()
    main(args)
