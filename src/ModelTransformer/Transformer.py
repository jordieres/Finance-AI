import argparse
import os
import random
import sys
import time
import numpy as np
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.metrics import mean_squared_error

sys.path.append('D:/Escritorio/TFG/Finance-AI/src')
from utils_vv_tfg import save_data, load_preprocessed_data, denormalize_data
from config.config import get_configuration

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Modify embedding layer
        self.attentions = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.feedforwards = nn.ModuleList([nn.Sequential(
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim)
                                            ) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)  # Output is a single value
    
    def forward(self, src, tgt=None):
        src_embedded = self.embedding(src)
        src_embedded = src_embedded.permute(1, 0, 2)  # Permute to match Transformer input format: (seq_len, win_size, hidden_dim)
        
        if tgt is not None:
            tgt_embedded = self.embedding(tgt)
            tgt_embedded = tgt_embedded.permute(1, 0, 2)
        
        for attention, feedforward in zip(self.attentions, self.feedforwards):
            if tgt is not None:
                attention_output, _ = attention(tgt_embedded, src_embedded, src_embedded)
                tgt_embedded = feedforward(attention_output)
            else:
                attention_output, _ = attention(src_embedded, src_embedded, src_embedded)
                src_embedded = feedforward(attention_output)
        
        if tgt is not None:
            output = self.output_layer(tgt_embedded[-1])  # Use only the last time step
        else:
            output = self.output_layer(src_embedded[-1])
        
        return output

def transformer_fun(transformer_parameters, trainX, trainY, testX, testY, Y, vdd, epochs, bsize, nhn, win, n_ftrs, ahead, stock, seed):
    # Convert to PyTorch tensors
    nptrX = torch.tensor(trainX.to_numpy().reshape(n_ftrs, trainX.shape[0], trainX.shape[1]), dtype=torch.float32)
    nptrY = torch.tensor(trainY.to_numpy(), dtype=torch.float32)
    nptstX = torch.tensor(testX.to_numpy().reshape(n_ftrs, testX.shape[0], testX.shape[1]), dtype=torch.float32)
    nptstY = torch.tensor(testY.to_numpy(), dtype=torch.float32)

    nit = 0
    lloss = float('nan')
    while torch.isnan(torch.Tensor([lloss])) and nit < 5:
        torch.manual_seed(seed)
        
        # Create the model
        model = Transformer(input_dim=(win*nptrX.shape[0]), hidden_dim=nhn, num_layers=transformer_parameters['num_layers'], num_heads=transformer_parameters['num_heads'], dropout=0.05)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        outputs = []
        # Train the model
        print("Training the model...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            for i in range(0, len(nptrX), bsize):
                batch_nptrX = nptrX[:, i:i+bsize, :]
                batch_nptrY = nptrY[i:i+bsize]
                output = model(batch_nptrX)
                loss = criterion(output, batch_nptrY.unsqueeze(1))  # Unsqueeze to match output shape
                loss.backward()
                optimizer.step()
                outputs.append(output.detach().numpy())
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epoch}]: Loss: {loss.item()}")

        lloss = loss.item()
        nit += 1

        jdx = testX.index
        Yf = denormalize_data(outputs, vdd, jdx) # denormalize the forecast
        Yf = Yf.apply(lambda x: x[0][0]) # get the value
        Yr  = Y.loc[jdx] # y real
        Yy  = Y.shift(ahead).loc[jdx] # Y yesterday
        DY  = pd.concat([Yr,Yf,Yy],axis=1)
        DY.columns = ['Y_real','Y_predicted','Y_yesterday']
        
    # Evaluate the model
    with torch.no_grad():
        outputs = model(nptstX)
    
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
    config, _ = get_configuration()
    processed_path = config['data']['output_path']
    multi = config['multi']
    
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
        tmod = config['model']    # lstm stcklstm or attlstm
        scen_model = f'MODEL_{scenarios[i]}'
        res[scen_model] = tmod
        num_layers = config['num_layers']
        num_heads = config['num_heads']
        transformer_parameters = {
            'num_layers': num_layers,
            'num_heads': num_heads
        }

        #for stock in stock_list:
        for stock in ['AAPL']:
            for tr_tst in list_tr_tst:
                _, _, lahead, lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, multi)
                win, n_ftrs, tr_tst, deep = lpar

                fmdls = f'D:/Escritorio/TFG/Finance-AI/Models/{nhn}{tmod}/{tr_tst}/{stock}/'
                if not os.path.exists(fmdls):
                    os.makedirs(fmdls)
                res[scenarios[i]] = {}
                res[scenarios[i]][stock] = {}
                print(f"Traning for {tr_tst*100}% training data")
                for ahead in lahead:
                    print('######################################################')
                    print('Training ' + stock + ' ahead ' + str(ahead) + ' days.')
                    trainX = tot_res['INPUT_DATA'][scenarios[i]][ahead]['trainX']
                    trainY = tot_res['INPUT_DATA'][scenarios[i]][ahead]['trainY']
                    testX  = tot_res['INPUT_DATA'][scenarios[i]][ahead]['testX']    
                    testY  = tot_res['INPUT_DATA'][scenarios[i]][ahead]['testY']
                    Y      = tot_res['INPUT_DATA'][scenarios[i]][ahead]['y']
                    vdd    = tot_res['INPUT_DATA'][scenarios[i]][ahead]['vdd']
                    tmpr = []
                    
                    for irp in range(n_itr):
                        seed      = random.randint(0,1000)
                        transformer_start= time.time()
                        mdl_name  = '{}-{}-{}-{:02}.hd5'.format(tmod,stock,ahead,irp)
                        if tmod == "transformer":
                            sol   = transformer_fun(transformer_parameters,trainX,trainY,testX,testY,Y,vdd,testY.shape[0],bsize,nhn,win,n_ftrs,ahead,stock,seed)
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
                    res[scenarios[i]][stock][ahead] = pd.DataFrame(tmpr)
                
                tot_res['OUT_MODEL'] = res
                
                fdat = f'D:/Escritorio/TFG/Finance-AI/DataProcessed/output/{win}/{tr_tst}/{stock}-{tmod}-{num_layers}-{num_heads}-output.pkl'
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