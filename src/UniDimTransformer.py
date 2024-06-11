'''This module trains a univariate Transformer model for time series forecasting'''
import os
import random
import sys
import time
import warnings
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils_vv_tfg import save_data, load_preprocessed_data, denormalize_data
from config.config import get_configuration

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class TransformerL(nn.Module):
    """
    Transformer-based model for time series forecasting.
    """
    def __init__(self, input_dim: int, embed_dim: int, num_layers: int, num_heads: int, dropout: float):
        super(TransformerL, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.embedding = nn.Linear(self.input_dim, self.embed_dim)        
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, dim_feedforward=4*self.embed_dim, dropout=self.dropout)
                                             for _ in range(self.num_layers)])
                                             
        self.output_layer = nn.Linear(self.embed_dim, 1)

    def forward(self, src) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_length, 1).
        """
        src_embedded = self.embedding(src)
        src_embedded = src_embedded.permute(1, 0, 2)

        for encoder in self.encoder_layers:
            src_embedded = encoder(src_embedded)

        #src_embedded = src_embedded.permute(1, 0, 2)  # Volvemos a la forma original (batch, seq, embed)
        output = self.output_layer(src_embedded)
        return output


def transformer_fun(transformer_parameters: dict, train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.DataFrame,
                    Y: pd.DataFrame, vdd: dict, epochs: int, bsize: int, nhn: int, win: int, n_ftrs: int, ahead: int, stock: str, seed: int) -> dict:
    """
    Trains a Transformer model for time series forecasting.

    Parameters
    ----------
    transformer_parameters : dict
        Dictionary containing the number of layers and heads for the Transformer model.
    train_X : pd.DataFrame
        Training input data.
    train_y : pd.DataFrame
        Training target data.
    test_X : pd.DataFrame
        Test input data.
    test_y : pd.DataFrame
        Test target data.
    Y : pd.DataFrame
        Target data.
    vdd : dict
        Dictionary containing the mean, min, and max values used for normalization.
    epochs : int
        Number of epochs.
    bsize : int
        Batch size.
    nhn : int
        Number of hidden neurons (embedding dimension).
    win : int
        Window size.
    n_ftrs : int
        Number of features.
    ahead : int
        Number of days ahead.
    stock : str
        Stock ticker.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing the results of the Transformer model training.
    """
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np_train_X = torch.tensor(train_X.to_numpy().reshape(train_X.shape[1], train_X.shape[0], n_ftrs),
                              dtype=torch.float32).to(device)
    np_train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float32).to(device)
    np_test_X = torch.tensor(test_X.to_numpy().reshape(test_X.shape[1], test_X.shape[0], n_ftrs), 
                             dtype=torch.float32).to(device)
    np_test_y = torch.tensor(test_y.to_numpy(), dtype=torch.float32).to(device)

    nit = 0
    lloss = float('nan')
    model = TransformerL(input_dim=(np_train_X.shape[2]), embed_dim=nhn, 
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
                batch_np_train_X = np_train_X[:, i:i+bsize, :].to(device)
                batch_np_train_y = np_train_y[i:i+bsize].to(device)
                output = model(batch_np_train_X)
                loss = criterion(output, batch_np_train_y)  # Unsqueeze to match output shape
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

    jdx = test_X.index
    y_forecast = denormalize_data(outputs[0], vdd, jdx, False) # denormalize the forecast
    y_real  = Y.loc[jdx] # y real
    y_yesterday  = Y.shift(ahead).loc[jdx] # Y yesterday
    DY  = pd.concat([y_real,y_forecast,y_yesterday],axis=1)
    DY.columns = ['Y_real','Y_predicted','Y_yesterday']  
    msep = mean_squared_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    msey = mean_squared_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real
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


def main(args) -> None:
    """
    Main function to train a Transformer model for time series forecasting.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    config, _ = get_configuration(args.params_file)
    processed_path = config['data']['output_path']
    multi = False
    transformer_configs = []
    scenarios = []
    for scenario in config['scenarios']:
        list_win_size = scenario['win']
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
        for i, config_transformer in enumerate(transformer_configs):
            tmod = config_transformer['model']
            scen_model = 'MODEL_'+scenario['name']
            res[scen_model] = tmod
            num_layers = config_transformer['num_layers']
            num_heads = config_transformer['num_heads']
            transformer_parameters = {
                'num_layers': num_layers,
                'num_heads': num_heads
            }

        for win_size in list_win_size:
            for tr_tst in list_tr_tst:
                out_model = {}
                for stock in stock_list:
                    lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, scenario['name'], multi)
                    win, n_ftrs, tr_tst = lpar
                    name = scenario['name']
                    fmdls = config['data']['fmdls'].format(nhn=nhn,tmod=tmod,tr_tst=tr_tst,stock=stock)
                    if not os.path.exists(fmdls):
                        os.makedirs(fmdls)
                    res[stock] = {}
                    print(f"Traning for {win} window size {tr_tst*100}% training data")
                    for ahead in lahead:
                        tot = tot_res['INPUT_DATA'][ahead]
                        train_X = tot['trainX']
                        train_y = tot['trainY']
                        test_X  = tot['testX']    
                        test_y  = tot['testY']
                        Y      = tot['y']
                        vdd    = tot['vdd']
                        tmpr = []                 
                        for irp in range(n_itr):
                            seed      = random.randint(0,1000)
                            print('######################################################')
                            print(f'{irp} Training {stock} {ahead} days ahead.')
                            transformer_start= time.time()
                            mdl_name  = f'{name}-{tmod}-{stock}-{ahead}-{irp}'
                            if tmod == "transformer":
                                sol   = transformer_fun(transformer_parameters,train_X,train_y,test_X,test_y,
                                                        Y,vdd,epochs,bsize,nhn,win,n_ftrs,ahead,stock,seed)
                            transformer_end  = time.time()
                            ttrain    = transformer_end - transformer_start
                            sol['ttrain'] = ttrain
                            sol['epochs']  = epochs
                            sol['bsize']  = bsize
                            sol['nhn']    = nhn
                            sol['win']    = win
                            sol['tr_tst'] = tr_tst
                            sol['transformer_parameters'] = transformer_parameters
                            model_json 	= {}
                            vdd_json = vdd.copy()
                            vdd_json.pop('mean')
                            min = vdd_json['min'].iloc[0]
                            max = vdd_json['max'].iloc[0]
                            #vdd_json = vdd_json.to_dict()
                            model_json['vdd'] = {}
                            model_json['vdd']['min'] = min
                            model_json['vdd']['max'] = max
                            with open(f"{fmdls}{mdl_name}.json","w") as json_file:
                                    json.dump(model_json, json_file)
                            torch.save(sol['model'], f"{fmdls}{mdl_name}.h5")
                            sol['model']  = fmdls+mdl_name
                            print('   Effort spent: ' + str(ttrain) +' s.')
                            sys.stdout.flush()
                            tmpr.append(sol)
                        res[stock][ahead] = pd.DataFrame(tmpr)
                    out_model[stock] = res[stock]
                    
                tot_res['OUT_MODEL'] = out_model              
                fdat = '{}/output/{}/{}/{}-{}-output.pkl'.format(processed_path,win,tr_tst,scenario['name'],tmod)
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