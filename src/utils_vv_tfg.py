import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def save_data(fich, out_path, lahead, lpar, tot_res):
    '''
    Saves the data to a pickle file
    '''
    with open(fich, 'wb') as file:
            pickle.dump(out_path, file)
            pickle.dump(fich, file)
            pickle.dump(lahead, file)
            pickle.dump(lpar, file)
            pickle.dump(tot_res, file)
    file.close()
    
def load_preprocessed_data(path, win, tr_tst, ticker, multi):
    if multi == True:
        fdat = path+ f"/{win}/{tr_tst}/{ticker}-m-input-output.pkl"
    else:
        fdat = path+ f"/{win}/{tr_tst}/{ticker}-input-output.pkl"

    if os.path.exists(fdat):
        with open(fdat, "rb") as openfile:
            path = pickle.load(openfile)
            fdat = pickle.load(openfile)
            lahead = pickle.load(openfile)
            lpar = pickle.load(openfile)
            tot_res = pickle.load(openfile)

        return path, fdat, lahead, lpar, tot_res
    else:
        raise FileNotFoundError("El archivo {} no existe.".format(fdat))


def denormalize_data(Yn, mvdd, idx):
    """
    Returns the denormalized target data.
    
    Arguments:
    Yn - normalized target data
    mvdd - dictionary containing mean, min, and max values used for normalization
    idx - index of the data
    """

    mnmX = mvdd["min"]
    mxmX = mvdd["max"]
    mnx = mvdd["mean"]
    
    denormalized_mYl = pd.Series(Yn, index=idx) * (mxmX - mnmX) + mnmX + mnx
    denormalized_mYl.dropna(inplace=True)
    return denormalized_mYl

def eval_lstm(nptstX, nptstY, testX, model, vdd, Y, ahead):
        Yhat   = model.predict(nptstX,verbose=0)
        tans   = Yhat.shape
        if len(tans) > 2 and tans[1] == 1:
            Yhat= np.concatenate(Yhat,axis=0)
        if len(tans) > 2 and tans[1] > 1:
            Yhat= [vk[0].tolist() for vk in Yhat]
        Yprd   = np.concatenate(Yhat,axis=0).tolist()
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

def load_output_preprocessed_data(path, win, tr_tst, stock, multi=False):
    all_results = {}

    # Directory containing the results files
    directory = f'D:/Escritorio/TFG/Finance-AI/DataProcessed/output/{win}/{tr_tst}/'

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
                    path      = pickle.load(openfile)
                    fdat      = pickle.load(openfile)
                    lahead    = pickle.load(openfile)
                    lpar      = pickle.load(openfile)
                    tot_res   = pickle.load(openfile)
                    results['path'] = path
                    results['fdat'] = fdat
                    results['lahead'] = lahead
                    results['lpar'] = lpar
                    results['tot_res'] = tot_res
                except EOFError:
                    break
            all_results[mdl] = results

    return all_results


def plot_res(ax, DY, msep, msey, stck, mdl, itr, ahead, tr_tst, num_heads=None, num_layers=None):
    ax.plot(DY.index, DY.Y_real, label="Real Values")
    ax.plot(DY.index, DY.Y_predicted, label=f"{mdl.upper()} predicted", color='orange')
    ax.plot(DY.index, DY.Y_yesterday, label="Historical", color='green')
    ax.set_title(f'{stck} - {mdl.upper()}, Predict {ahead} days ahead. Iter: {itr},\n Heads: {num_heads}, Hidden Layers: {num_layers},\n Train: {tr_tst*100}%')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    # Agregar texto
    ax.text(.01, .01, 'MSE Predicted=' + str(round(msep,3)), transform=ax.transAxes, ha='left', va='bottom', fontsize=16, color='#FFA500')
    ax.text(.01, .05, 'MSE Historical=' + str(round(msey,3)), transform=ax.transAxes, fontsize=16, ha='left', va='bottom', color='green')


def plot_results_comparison(model_results, model_list, stck, itr, ahead, tr_tst, save_path=None):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(model_list):
        col = i % 3
        res = model_results[model]['tot_res']['OUT_MODEL'][ahead]
        DYs = res['DY']
        DY = DYs.loc[itr]
        msep = res['MSEP'][itr]
        msey = res['MSEY'][itr]
        
        if model != 'transformer':
            plot_res(axs[col], DY, msep, msey, stck, model, itr, ahead, tr_tst)
        else:
            num_heads = res['transformer_parameters'][0]['num_heads']
            num_layers = res['transformer_parameters'][0]['num_layers']
            plot_res(axs[col], DY, msep, msey, stck, model, itr, ahead, tr_tst, num_heads, num_layers)
        
        # Ajustar los límites del eje X y Y
        axs[col].set_xlim(DY.index.min(), DY.index.max())  # Ajustar los límites del eje X
        axs[col].set_ylim(120, 200)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.tight_layout()
        figfich = os.path.join(save_path, f'{stck}-{itr}-{num_heads}-{num_layers}.png')
        plt.savefig(figfich)
    else:
        plt.tight_layout()
        plt.show()