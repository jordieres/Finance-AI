import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def save_data(fich, out_path, lahead, lpar, tot_res):
    '''Saves the data to a pickle file
    ...
    Parameters
    ----------
    fich : str
        Path to the file where the data will be saved
    out_path : str
        Path to the output file
    lahead : list
        List of the number of days ahead to forecast
    lpar : list
        List of the parameters used in the model
    tot_res : dict
        Dictionary containing the results of the model
    '''
    with open(fich, 'wb') as file:
        pickle.dump(out_path, file)
        pickle.dump(fich, file)
        pickle.dump(lahead, file)
        pickle.dump(lpar, file)
        pickle.dump(tot_res, file)
    file.close()
    
def load_preprocessed_data(path, win, tr_tst, ticker, scen_name, multi):
    '''Loads the preprocessed data from a pickle file
    ...
    Parameters
    ----------
    path : str
        Path to the file where the data is saved
    win : int
        Number of days to consider in the window
    tr_tst : float
        Percentage of the data to use for training
    ticker : str
        Ticker of the stock
    multi : bool
        True if the model is a multi-input model, False otherwise
    
    Returns
    -------
    path : str
        Path to the file where the data is saved
    fdat : str
        Path to the data file
    lahead : list
        List of the number of days ahead to forecast
    lpar : list
        List of the parameters used in the model
    tot_res : dict
        Dictionary containing the results of the model
        '''
    if multi is True:
        fdat = path+ f"/input/{win}/{tr_tst}/{scen_name}-{ticker}-m-input.pkl"
    else:
        fdat = path+ f"/input/{win}/{tr_tst}/{scen_name}-{ticker}-input.pkl"

    if os.path.exists(fdat):
        with open(fdat, "rb") as openfile:
            lpath = pickle.load(openfile)
            lfdat = pickle.load(openfile)
            lahead = pickle.load(openfile)
            lparams = pickle.load(openfile)
            ltot_res = pickle.load(openfile)
        return lparams, ltot_res
    else:
        raise FileNotFoundError(f"El archivo {fdat} no existe.")
    
def select_features(features, selected_features):
    features_index = []
    for i, feature in enumerate(features):
        if feature in selected_features:
            features_index.append(i)
    return features_index

def denormalize_data(Yn, mvdd, idx, multi, lstm=False):
    """
    Returns the denormalized target data.
    ...
    Parameters
    ----------
    Yn : np.array
        Normalized target data
    mvdd : dict
        Dictionary containing the mean, min and max values of the target data
    idx : pd.Index
        Index of the target data
    multi : bool
        True if the model is a multi-input model, False otherwise

    Returns
    -------
    denormalized_Yn : pd.Series
        Denormalized target data    
    """
    if lstm is True:    
        min_x = mvdd["min"]
        max_x = mvdd["max"]
        mean_x = mvdd["mean"]
        denormalized_Yn = pd.Series(Yn, index=idx) * (max_x - min_x) + min_x + mean_x
        denormalized_Yn.dropna(inplace=True)
    
    else:
        if multi is False:
            min_x = mvdd["min"]
            max_x = mvdd["max"]
            mean_x = mvdd["mean"]
            denormalized_Yn = pd.Series(Yn[:,0,0], index=idx) * (max_x - min_x) + min_x + mean_x
            denormalized_Yn.dropna(inplace=True)
        else:
            min_x = mvdd["min"][0]
            max_x = mvdd["max"][0]
            mean_x = mvdd["mean"]['PX_LAST']
            denormalized_Yn = pd.Series(Yn[:,0,0], index=idx) * (max_x - min_x) + min_x + mean_x
            denormalized_Yn.dropna(inplace=True)
    
    return denormalized_Yn

def eval_lstm(np_test_X, np_test_y, test_X, model, vdd, Y, ahead):
    '''Evaluates the LSTM model
    ...
    Parameters
    ----------
    np_test_X : np.array
        Test input data
    np_test_y : np.array
        Test target data
    test_X : pd.DataFrame
        Test input data
    model : keras model
        LSTM model
    vdd : dict
        Dictionary containing the mean, min and max values of the target data
    Y : pd.Series
        Target data
    ahead : int
        Number of days ahead to forecast

    Returns
    -------
    dict
        Dictionary containing the mean squared error of the predicted and historical values,
        the predicted and real values, and the historical values
        '''
    y_hat   = model.predict(np_test_X,verbose=0)
    tans   = y_hat.shape
    if len(tans) > 2 and tans[1] == 1:
        y_hat= np.concatenate(y_hat,axis=0)
    if len(tans) > 2 and tans[1] > 1:
        y_hat= [vk[0].tolist() for vk in y_hat]
    y_pred   = np.concatenate(y_hat,axis=0).tolist()
    eff    = pd.DataFrame({'Yorg':np_test_y,'Yfct':y_pred}) 
    eff.set_index(test_X.index,drop=True,inplace=True)
    absolute_difference = np.abs(y_pred - np_test_y)
    correct = np.count_nonzero(absolute_difference <= 0.1)
    accuracy = round((correct / len(np_test_y)) *100, 3)
    print(f"Accuracy: {accuracy}%")
    jdx = test_X.index
    y_forecast = denormalize_data(y_pred, vdd, jdx, multi=False, lstm=True) # denormalize the forecast
    y_real  = Y.loc[jdx] # y real
    y_yesterday  = Y.shift(ahead).loc[jdx] # Y yesterday
    DY  = pd.concat([y_real,y_forecast,y_yesterday],axis=1)
    DY.columns = ['Y_real','Y_predicted','Y_yesterday']

    msep= mean_squared_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    msey= mean_squared_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real
    maep = mean_absolute_error(DY.Y_predicted , DY.Y_real) # error y predicted - y real
    maey = mean_absolute_error(DY.Y_yesterday , DY.Y_real) # error y yesterday - y real

    return({'msep':msep,'msey':msey, 'maep':maep, 'maey':maey, 'Ys':DY, 'eff': eff})

def load_output_preprocessed_data(win, tr_tst, multi):
    '''Loads the preprocessed data from the output files
    ...
    Parameters
    ----------
    win : int
        Number of days to consider in the window
    tr_tst : float
        Percentage of the data to use for training
    multi : bool
        True if the model is a multi-input model, False otherwise

    Returns
    -------
    all_results : dict
        Dictionary containing the results of the models
    '''
    all_results = {}

    # Directory containing the results files
    directory = f'home/vvallejo/Finance-AI/dataprocessed/output/{win}/{tr_tst}/'

    # Get list of files in directory
    files = os.listdir(directory) 

    for file in files: # Loop through each file in the directory
        # Verify if the file is type 'output.pkl' o 'output-m.pkl'
        if file.endswith('-output.pkl') and not multi:
            mdl = file.split('-')[1]
        elif file.endswith('-m-output.pkl') and multi:
            mdl = file.split('-')[1]
        else:
            continue
        
        # Construir la ruta completa del archivo
        fdat = os.path.join(directory, file)

        with (open(fdat, "rb")) as openfile:
            results = {}
            while True:
                try:
                    directory1      = pickle.load(openfile)
                    fdat1      = pickle.load(openfile)
                    lahead    = pickle.load(openfile)
                    lpar      = pickle.load(openfile)
                    tot_res   = pickle.load(openfile)
                    results['path'] = directory1
                    results['fdat'] = fdat1
                    results['lahead'] = lahead
                    results['lpar'] = lpar
                    results['tot_res'] = tot_res
                except EOFError:
                    break
            all_results[mdl] = results

    return all_results


def plot_res(ax, DY, msep, msey, stck, mdl, itr, ahead, tr_tst, num_heads=None, num_layers=None):
    '''Plots the results of the model
    ...
    Parameters
    ----------  
    ax : matplotlib.axes.Axes
        Axes object
    DY : pd.DataFrame
        Dataframe containing the real, predicted and historical values
    msep : float
        Mean squared error of the predicted and real values
    msey : float
        Mean squared error of the historical and real values
    stck : str
        Ticker of the stock
    mdl : str
        Name of the model
    itr : int
        Number of iterations
    ahead : int
        Number of days ahead to forecast
    tr_tst : float
        Percentage of the data to use for training
    num_heads : int
        Number of heads in the transformer model
    num_layers : int
        Number of layers in the transformer model
        '''
    ax.plot(DY.index, DY.Y_real, label="Real Values")
    ax.plot(DY.index, DY.Y_predicted, label=f"{mdl.upper()} predicted", color='orange')
    ax.plot(DY.index, DY.Y_yesterday, label="Historical", color='green')
    ax.set_title(f'{stck} - {mdl.upper()}, Predict {ahead} days ahead. Iter: {itr},\n \
                Heads: {num_heads}, Hidden Layers: {num_layers},\n Train: {tr_tst*100}%')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    # Agregar texto
    ax.text(.01, .01, 'MSE Predicted=' + str(round(msep,3)), transform=ax.transAxes, ha='left', va='bottom', fontsize=16, color='#FFA500')
    ax.text(.01, .05, 'MSE Historical=' + str(round(msey,3)), transform=ax.transAxes, fontsize=16, ha='left', va='bottom', color='green')


def plot_results_comparison(model_results, lahead, model_list, scen_name, stck, itr, tr_tst, save_path=None, wdth=10, hght=30):
    '''Plots the results of the models
    ...
    Parameters
    ----------
    model_results : dict
        Dictionary containing the results of the models
    lahead : list
        List of the number of days ahead to forecast
    model_list : list
        List of the models to compare
    scen_name : str
        Name of the scenario
    stck : str
        Ticker of the stock
    itr : int
        Number of iterations
    tr_tst : float
        Percentage of the data to use for training
    save_path : str
        Path to save the figure
    wdth : int
        Width of the figure
    hght : int
        Height of the figure
        '''
    num_models = len(model_list)
    fig, axs = plt.subplots(len(lahead), len(model_list), figsize=[wdth*num_models, hght])
    
    for j, ahead in enumerate(lahead):
        for i, model in enumerate(model_list):
            col = i
            res = model_results[model]['tot_res']['OUT_MODEL'][scen_name][stck][ahead]
            DYs = res['DY']
            DY = DYs.loc[itr]
            msep = res['MSEP'][itr]
            msey = res['MSEY'][itr]
            
            if model != 'transformer':
                plot_res(axs[j, col], DY, msep, msey, stck, model, itr, ahead, tr_tst)
            else:
                num_heads = res['transformer_parameters'][0]['num_heads']
                num_layers = res['transformer_parameters'][0]['num_layers']
                plot_res(axs[j, col], DY, msep, msey, stck, model, itr, ahead, tr_tst, num_heads, num_layers)               
            # Ajustar los límites del eje X y Y
            axs[j, col].set_xlim(DY.index.min(), DY.index.max())  # Ajustar los límites del eje X
            axs[j, col].set_ylim(120, 200)    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.tight_layout()
        figfich = os.path.join(save_path, f'{stck}-{itr}-{num_heads}-{num_layers}.png')
        plt.savefig(figfich)
    else:
        plt.tight_layout()
        plt.show()

def plot_ahead_perf_sameStock(model_results, lahead, model_list, stock_list, list_tr_tst, scen_name, wdth=8, hght=9):
    '''Plots the performance of the models for the same stock
    ...
    Parameters
    ----------
    model_results : dict
        Dictionary containing the results of the models
    lahead : list
        List of the number of days ahead to forecast
    model_list : list
        List of the models to compare
    stock_list : list
        List of the stocks to compare
    list_tr_tst : list
        List of the percentages of the data to use for training
    scen_name : str
        Name of the scenario
    wdth : int
        Width of the figure
    hght : int
        Height of the figure    
        '''
    num_models = len(model_list)
    fig, axes = plt.subplots(len(list_tr_tst), num_models, figsize=[wdth*num_models, hght])

    for stock in stock_list:
        for i, tr_tst in enumerate(list_tr_tst):
            for j, model in enumerate(model_list):
                res = model_results[model]['tot_res']['OUT_MODEL'][scen_name][stock]
                lstd = {}
                yval = {}
                for ahead in lahead:
                    lstd[ahead] = res[ahead]['MSEP']
                    yval[ahead] = res[ahead]['MSEY'][0]

                h = list(range(len(lstd)))
                bp = np.array(list(lstd.values()))

                ax = axes[i, j]  # Acceder al eje en la posición i, j
                ax.boxplot(bp.T, positions=h, showmeans=True, manage_ticks=False)
                ax.plot(h, yval.values(), '--ko', c='red', label='Yesterday')
                ax.set_xticks(h)
                ax.set_xticklabels(list(lstd.keys()), rotation='vertical')
                ax.set_xlabel('Stocks')
                ax.set_ylabel('MSE of the 10 simulations')
                ax.set_yscale('log')
                ax.set_title(f'{model.upper()} for {stock} and {tr_tst*100}% train')
                ax.legend()

        plt.tight_layout()
        plt.show()