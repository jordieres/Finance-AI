import os
import random
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error


sys.path.append('/home/vvallejo/Finance-AI/src')
from utils_vv_tfg import save_data, load_preprocessed_data, denormalize_data, select_features
from config.config import get_configuration

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.embedding = nn.Linear(self.input_dim, self.embed_dim)        
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, dim_feedforward=4*self.embed_dim, dropout=self.dropout)
                                             for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.embed_dim, 1)

    def forward(self, src):
        src_embedded = self.embedding(src)
        src_embedded = src_embedded.permute(1, 0, 2)

        for encoder in self.encoder_layers:
            src_embedded = encoder(src_embedded)

        #src_embedded = src_embedded.permute(1, 0, 2)  # Volvemos a la forma original (batch, seq, embed)
        output = self.output_layer(src_embedded)
        return output

def transformer_fun(transformer_parameters, train_X, train_y, test_X, test_y, 
                    Y, vdd, epochs, bsize, nhn, win, n_ftrs, ahead, stock, test_X_idx, k, seed):
    '''Trains a Transformer model for time series forecasting
    ...
    Parameters
    ----------
    transformer_parameters : dict
        dictionary containing the number of layers and heads for the Transformer model
    train_X : pd.DataFrame
        training input data
    train_y : pd.DataFrame
        training target data
    test_X : pd.DataFrame
        test input data
    test_y : pd.DataFrame
        test target data
    Y : pd.DataFrame
        target data
    vdd : dict
        dictionary containing the mean, min, and max values used for normalization
    epochs : int
        number of epochs
    bsize : int
        batch size
    nhn : int
        number of hidden neurons (embedding dimension)
    win : int
        window size
    n_ftrs : int
        number of features
    ahead : int
        number of days ahead
    stock : str
        stock ticker
    seed : int
        random seed

    Returns
    -------
    df_result
        dictionary containing the results of the Transformer model training
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert to PyTorch tensors
    np_train_X = torch.tensor(train_X.reshape(train_X.shape[1], train_X.shape[0], k), 
                              dtype=torch.float32).to(device)
    np_train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    np_test_X = torch.tensor(test_X.reshape(test_X.shape[1], test_X.shape[0], k), 
                             dtype=torch.float32).to(device)
    np_test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    nit = 0
    lloss = float('nan')
    model = Transformer(input_dim=k, embed_dim=nhn, 
                        num_layers=transformer_parameters['num_layers'], num_heads=transformer_parameters['num_heads'], dropout=0.01)
    # Define the loss function and optimizer
    model.to(device)
    while torch.isnan(torch.Tensor([lloss])) and nit < 5:
        torch.manual_seed(seed)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)       
        outputs = []
        # Train the model
        print("Training the model...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            for i in range(0, len(np_train_X), bsize):
                batch_np_train_X = np_train_X[:, i:i+bsize, :].to(device) # (window size, batch size, number of features)
                batch_np_train_y = np_train_y[i:i+bsize].to(device)
                output = model(batch_np_train_X)
                loss = criterion(output, batch_np_train_y)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]:")
                print(f"Train Loss: {loss.item()}")
            #Evaluate the model
            with torch.no_grad():
                y_hat  = model(np_test_X)
                test_loss = criterion(y_hat, np_test_y)
            if epoch % 10 == 0:
                print(f"Test Loss: {test_loss.item()}")
                print('-----------------------------------')
        outputs.append(y_hat.cpu().detach().numpy())
        lloss = loss.item()
        nit += 1
        
    y_forecast = denormalize_data(outputs[0], vdd, test_X_idx, True)
    #y_forecast = y_forecast.apply(lambda x: x) # get the value
    y_real  = Y.loc[test_X_idx] # y real
    y_yesterday  = Y.shift(ahead).loc[test_X_idx] # Y yesterday
    DY  = pd.concat([y_real,y_forecast,y_yesterday],axis=1)
    DY.columns = ['Y_real','Y_predicted','Y_yesterday']
    
    msep = mean_squared_error(DY.Y_predicted, DY.Y_real) # error y predicted - y real
    msey = mean_squared_error(DY.Y_yesterday, DY.Y_real) # error y yesterday - y real
    maep = mean_absolute_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    maey = mean_absolute_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real
    # Prepare the result dictionary
    df_result = {
        'MSEP': msep,'MSEY': msey,
        'MAEP': maep, 'MAEY': maey, 
        'Stock': stock,'DY': DY,
        'ALG': 'Transformer','seed': seed,
        'epochs': epoch,'nhn': nhn,'win': win,
        'ndims': 1,'lossh': lloss,
        'nit': nit,'model': model
    }

    return df_result


def main():
    '''Main function to train a Transformer model for time series forecasting'''
    config, _ = get_configuration()
    processed_path = config['data']['output_path']
    multi = True
    transformer_configs = []
    scenarios = []
    for scenario in config['scenarios']:
        win_size = scenario['win']
        list_tr_tst = scenario['tr_tst']
        lahead = scenario['lahead']
        epochs = scenario['epochs']
        bsize = scenario['batch_size']
        nhn = scenario['nhn']
        n_itr = scenario['n_itr']
        stock_list = scenario['tickers']
        if 'Transformer' in scenario:
            transformer_configs.append(scenario['Transformer'])
            scenarios.append(scenario['name'])
        res = {}
        for i, config in enumerate(transformer_configs):
            tmod = config['model']
            scen_model = 'MODEL_'+scenario['name']
            res[scen_model] = tmod
            num_layers = config['num_layers']
            num_heads = config['num_heads']
            scenario_features = config['features']
            k_variables = len(scenario_features)
            transformer_parameters = {
                'num_layers': num_layers,
                'num_heads': num_heads
            }

        for tr_tst in list_tr_tst:
            out_model = {}
            transformer_parameters['num_variables'] = k_variables
            for stock in stock_list:               
                lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, scenario['name'], multi)
                win, n_ftrs, tr_tst = lpar
                fmdls = f'/home/vvallejo/Finance-AI/Models/{nhn}{tmod}/{tr_tst}/{stock}/'
                if not os.path.exists(fmdls):
                    os.makedirs(fmdls)
                res[stock] = {}
                print(f"Traning for {tr_tst*100}% training data")
                for ahead in lahead:
                    tot = tot_res['INPUT_DATA'][ahead]
                    train_X = tot['trainX']
                    train_y = tot['trainY']
                    test_X  = tot['testX']    
                    test_y  = tot['testY']
                    Y      = tot['y']
                    vdd    = tot['vdd']
                    test_X_idx = tot['idtest']
                    features = tot['cnms']
                    selected_features = select_features(features, scenario_features)
                    train_X = train_X[:,:,selected_features]
                    test_X = test_X[:,:,selected_features]
                    tmpr = []                  
                    for irp in range(n_itr):
                        seed      = random.randint(0,1000)
                        print('######################################################')
                        print(f'{irp} Training {stock} {ahead} days ahead')
                        transformer_start= time.time()
                        name = scenario['name']
                        mdl_name  = f'{name}-{tmod}-{stock}-{ahead}-{irp}.hd5'
                        if tmod == "transformer":
                            sol   = transformer_fun(transformer_parameters,train_X,train_y,test_X,test_y,
                                                    Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,test_X_idx,k_variables,seed)
                        transformer_end  = time.time()
                        ttrain    = transformer_end - transformer_start
                        sol['ttrain'] = ttrain
                        sol['epochs']  = epochs
                        sol['bsize']  = bsize
                        sol['nhn']    = nhn
                        sol['win']    = win
                        sol['tr_tst'] = tr_tst
                        sol['transformer_parameters'] = transformer_parameters
                        sol['model']  = fmdls+mdl_name
                        print('   Effort spent: ' + str(ttrain) +' s.')
                        sys.stdout.flush()
                        tmpr.append(sol)
                    res[stock][ahead] = pd.DataFrame(tmpr)
                out_model[stock] = res[stock]

            tot_res['OUT_MODEL'] = out_model    
            fdat = '/home/vvallejo/Finance-AI/dataprocessed/output/{}/{}/{}-{}-m-output.pkl'.format(win,tr_tst,scenario['name'],tmod)
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
    main()