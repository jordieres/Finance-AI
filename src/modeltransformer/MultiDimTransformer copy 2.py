'''This module trains a univariate Transformer model for time series forecasting'''
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
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest

sys.path.append('/home/vvallejo/Finance-AI/src')
from utils_vv_tfg import save_data, load_preprocessed_data, denormalize_data
from config.config import get_configuration

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.fc2(torch.relu(self.fc1(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, dropout=self.dropout) 
                                            for _ in range(self.num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(self.embed_dim, self.num_heads, dropout=self.dropout)
                                            for _ in range(self.num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.feedforwards = nn.ModuleList([nn.Sequential(
                                            nn.Linear(self.embed_dim, 4 * self.embed_dim),
                                            nn.ReLU(),
                                            nn.Linear(4 * self.embed_dim, self.embed_dim)
                                            ) for _ in range(self.num_layers)])
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.output_layer = nn.Linear(self.embed_dim, 1)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src.view(-1, src.shape[-1]))
        src_embedded = self.positional_encoding(src_embedded)
        src_embedded = src_embedded.permute(1, 0, 2)
        
        for layer in self.encoder_layers:
            src_embedded = self.layer_norm1(src_embedded)
            src_embedded = layer(src_embedded)
            src_embedded = self.layer_norm2(src_embedded)

        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)
        
        for layer in self.decoder_layers:
            tgt_embedded = self.layer_norm1(tgt_embedded)
            tgt_embedded = layer(tgt_embedded, src_embedded)
            tgt_embedded = self.layer_norm2(tgt_embedded)

        output = self.output_layer(tgt_embedded)
        output = self.dropout(output)
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
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np_train_X = torch.tensor(train_X.reshape(train_X.shape[0], train_X.shape[1], k),
                              dtype=torch.float32).to(device)
    np_train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    np_test_X = torch.tensor(test_X.reshape(test_X.shape[0], test_X.shape[1], k),
                             dtype=torch.float32).to(device)
    np_test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    nit = 0
    lloss = float('nan')
    model = Transformer(input_dim=k, embed_dim=nhn, 
                        num_layers=transformer_parameters['num_layers'], num_heads=transformer_parameters['num_heads'], dropout=0.05)
    model.to(device)
    while torch.isnan(torch.Tensor([lloss])) and nit < 5:
        torch.manual_seed(seed)
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        outputs = []
        # Train the model
        print("Training the model...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            for i in range(0, len(np_train_X), bsize):
                batch_np_train_X = np_train_X[i:i+bsize, :, :].to(device)
                batch_np_train_y = np_train_y[i:i+bsize].to(device)
                output = model(batch_np_train_X, batch_np_train_X)  # Pass the same input as both encoder and decoder
                loss = criterion(output, batch_np_train_y)  # Unsqueeze to match output shape
                
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]: Loss: {loss.item()}")
            with torch.no_grad():
                y_hat = model(np_test_X, np_test_X)  # Pass the same input as both encoder and decoder
                absolute_difference = np.abs(y_hat[:,0,0].cpu().detach().numpy() - np_test_y.cpu().detach().numpy())
                correct = np.count_nonzero(absolute_difference <= 0.1)
                acc = round((correct / len(np_test_y)) *100, 3)
            if epoch % 10 == 0:
                print(f'Accuracy: {acc}%')
        outputs.append(y_hat.cpu().detach().numpy())
        lloss = loss.item()
        nit += 1

    y_forecast = denormalize_data(outputs[0], vdd, test_X_idx, True) # denormalize the forecast
    y_forecast = y_forecast.apply(lambda x: x) # get the value
    y_real  = Y.loc[test_X_idx] # y real
    y_yesterday  = Y.shift(ahead).loc[test_X_idx] # Y yesterday
    DY  = pd.concat([y_real,y_forecast,y_yesterday],axis=1)
    DY.columns = ['Y_real','Y_predicted','Y_yesterday']  
    msep = mean_squared_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    msey = mean_squared_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real
    # Prepare the result dictionary
    df_result = {
        'MSEP': msep,'MSEY': msey,'Stock': stock,
        'DY': DY,'ALG': 'Transformer','seed': seed,'epochs': epoch,
        'nhn': nhn,'win': win,'ndims': 1,'lossh': lloss,
        'nit': nit,'model': model
    }

    return df_result



def main():
    '''Main function to train a Transformer model for time series forecasting'''
    config, _ = get_configuration()
    processed_path = config['data']['output_path']
    multi = False
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
            transformer_parameters = {
                'num_layers': num_layers,
                'num_heads': num_heads
            }

        for tr_tst in list_tr_tst:
            out_model_k = {}
            for k in [5]:
                transformer_parameters['num_variables'] = k
                out_model_stock = {}
                for stock in ['AMZN']:               
                    lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, True)
                    win, n_ftrs, tr_tst = lpar

                    fmdls = f'/home/vvallejo/Finance-AI/Models/{nhn}{tmod}/{tr_tst}/{stock}/'
                    if not os.path.exists(fmdls):
                        os.makedirs(fmdls)
                    res[scenario['name']] = {}
                    res[scenario['name']][stock] = {}
                    print(f"Traning for {tr_tst*100}% training data")
                    for ahead in [90]:
                        tot = tot_res['INPUT_DATA'][scenario['name']][ahead]
                        train_X = tot['trainX']
                        train_y = tot['trainY']
                        test_X  = tot['testX']    
                        test_y  = tot['testY']
                        Y      = tot['y']
                        vdd    = tot['vdd']
                        test_X_idx = tot['idtest']
                        X_train_reshaped = train_X[:, 0, :]
                        selector = SelectKBest(k=k)
                        selector.fit_transform(X_train_reshaped, train_y)
                        selected_features = list(selector.get_support(indices=True))
                        train_X = train_X[:,:,selected_features]
                        test_X = test_X[:,:,selected_features]
                        tmpr = []                  
                        for irp in range(n_itr):
                            seed      = random.randint(0,1000)
                            print('######################################################')
                            print(f'{irp} Training {stock} {ahead} days ahead. K={k}')
                            transformer_start= time.time()
                            mdl_name  = f'{tmod}-{stock}-{ahead}-{irp}_{k}.hd5'
                            if tmod == "transformer":
                                sol   = transformer_fun(transformer_parameters,train_X,train_y,test_X,test_y,
                                                        Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,test_X_idx,k,seed)
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
                        res[scenario['name']][stock][ahead] = pd.DataFrame(tmpr)

                    if scenario['name'] not in out_model_stock:
                        out_model_stock[scenario['name']] = {}
                    out_model_stock[scenario['name']][stock] = res[scenario['name']][stock]
                if scenario['name'] not in out_model_k:
                    out_model_k[scenario['name']] = {}
                '''if k not in out_model_k[scenario['name']].keys():
                    out_model_k[scenario['name']][k] = {}'''
                out_model_k[scenario['name']][k] = out_model_stock[scenario['name']]

            tot_res['OUT_MODEL'] = out_model_k     
            fdat = f'/home/vvallejo/Finance-AI/dataprocessed/output/{win}/{tr_tst}/prueba-{tmod}-m-output.pkl'
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