import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def save_data(fich: str, out_path: str, lahead: list, lpar: list, tot_res: dict) -> None:
    """Saves the data to a pickle file"""
    with open(fich, 'wb') as file:
        pickle.dump(out_path, file)
        pickle.dump(fich, file)
        pickle.dump(lahead, file)
        pickle.dump(lpar, file)
        pickle.dump(tot_res, file)
    
def load_preprocessed_data(path: str, win: int, tr_tst: float, ticker: str, scen_name: str, multi: bool) -> tuple:
    """Loads the preprocessed data from a pickle file"""
    fdat = f"{path}/input/{win}/{tr_tst}/{scen_name}-{ticker}-{'m-' if multi else ''}input.pkl"

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
    
def select_features(features: list, selected_features: list) -> list:
    """Selects the indices of the selected features"""
    return [i for i, feature in enumerate(features) if feature in selected_features]


def denormalize_data(Yn: np.array, mvdd: dict, idx: pd.Index, multi: bool, lstm: bool = False) -> pd.Series:
    """Returns the denormalized target data."""
    if lstm:
        min_x, max_x, mean_x = mvdd["min"], mvdd["max"], mvdd["mean"]
        denormalized_Yn = pd.Series(Yn, index=idx) * (max_x - min_x) + min_x + mean_x
    else:
        if not multi:
            min_x, max_x, mean_x = mvdd["min"], mvdd["max"], mvdd["mean"]
            denormalized_Yn = pd.Series(Yn[:, 0, 0], index=idx) * (max_x - min_x) + min_x + mean_x
        else:
            min_x, max_x, mean_x = mvdd["min"][0], mvdd["max"][0], mvdd["mean"]['PX_LAST']
            denormalized_Yn = pd.Series(Yn[:, 0, 0], index=idx) * (max_x - min_x) + min_x + mean_x

    denormalized_Yn.dropna(inplace=True)
    return denormalized_Yn

def eval_lstm(np_test_X: np.array, np_test_y: np.array, test_X: pd.DataFrame, model, vdd: dict, Y: pd.Series, ahead: int) -> dict:
    """Evaluates the LSTM model"""
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

def load_output_preprocessed_data(path: str, win: int, tr_tst: float, scenario_name: str) -> dict:
    """Loads all pickle files from the specified directory that match the scenario name"""
    all_results = {}
    directory = f'{path}/output/{win}/{tr_tst}/'

    # Verify if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    for file in os.listdir(directory):  # Loop through each file in the directory
        # Verify if the file starts with scenario_name and ends with 'output.pkl' or 'm-output.pkl'
        if file.startswith(scenario_name) and file.endswith('-output.pkl'):
            # Extract the model type correctly
            mdl = file.split('-')[1]
            if file.endswith('-m-output.pkl'):
                mdl = file.split('-')[1] + '-m'
        else:
            continue

        # Construct the full file path
        fdat = os.path.join(directory, file)

        with open(fdat, "rb") as openfile:
            results = {}
            while True:
                try:
                    directory1 = pickle.load(openfile)
                    fdat1 = pickle.load(openfile)
                    lahead = pickle.load(openfile)
                    lpar = pickle.load(openfile)
                    tot_res = pickle.load(openfile)
                    results['path'] = directory1
                    results['fdat'] = fdat1
                    results['lahead'] = lahead
                    results['lpar'] = lpar
                    results['tot_res'] = tot_res
                except EOFError:
                    break
            all_results[mdl] = results

    return all_results

def plot_res(ax, DY: pd.DataFrame, metric: str, metricp: float, metricy: float, ahead: int, *parameters: dict) -> None:
    ax.plot(DY.index, DY.Y_real, label="Real Values")
    p2 = ax.plot(DY.index, DY.Y_predicted, label="Transformer predicted", color='orange')
    p3 = ax.plot(DY.index, DY.Y_yesterday, label="Historical", color='green')
    if parameters:
        num_heads = parameters[0]['num_heads']
        num_layers = parameters[0]['num_layers']
        ax.set_title(f'Ahead: {ahead} days - {num_heads} heads - {num_layers} layers')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title(f'Ahead: {ahead} days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    # Agregar texto
    ax.text(.01, .01, f'{metric.upper()} Predicted=' + str(round(metricp,3)), transform=ax.transAxes, ha='left', va='bottom', fontsize=16, color='#FFA500')
    ax.text(.01, .05, f'{metric.upper()} Historical=' + str(round(metricy,3)), transform=ax.transAxes, fontsize=16, ha='left', va='bottom', color='green')

def run_plot_res(list_tr_tst: list, all_results: dict, stock_list: list, lahead: list, selected_scenario: str, metric: str, scen_name: str, plot_path: str, plot_format: str) -> None:
    for tr_tst in list_tr_tst:
        for model in all_results[tr_tst].keys():
            tot_res = all_results[tr_tst][model]['tot_res']
            for stock in stock_list:
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{stock} - {model.upper()} - {selected_scenario} - {metric} - {tr_tst*100}%', fontsize=16)
                for i, ahead in enumerate(lahead):
                    res1 = tot_res['OUT_MODEL'][stock]
                    itr = len(res1[ahead]['nit']) - 1
                    DYs = res1[ahead]['DY']
                    DY = DYs.loc[itr]
                    metricp = res1[ahead][f'{metric.upper()}P'][itr]
                    metricy = res1[ahead][f'{metric.upper()}Y'][itr]
                    row = i // 3
                    col = i % 3
                    if 'transformer' not in model:
                        plot_res(axs[row, col], DY, metric, metricp, metricy, ahead)
                    else:
                        plot_res(axs[row, col], DY, metric, metricp, metricy, ahead, res1[ahead]['transformer_parameters'][0])

                    axs[row, col].set_xlim(DY.index.min(), DY.index.max())  # Ajustar los límites del eje X
                    axs[row, col].set_ylim(DY['Y_yesterday'].min()*0.7, DY['Y_yesterday'].max()*1.3)
                path = f'{plot_path}/{scen_name}/{tr_tst}/'
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.tight_layout()
                figfich = path + f'{model}-{stock}-{itr}-{metric}.{plot_format}'
                plt.savefig(figfich)

def plot_metric_boxplots(selected_model: str, list_tr_tst: list, all_results: dict, stocks: list, lahead: list, metric: str, scen_name: str, plot_path: str, plot_format: str) -> None:
    mse_data = {stock: [] for stock in stocks}

    for tr_tst in list_tr_tst:
        tot_res = all_results[tr_tst][selected_model]['tot_res']
        for stock in stocks:
            for ahead in lahead:
                res1 = tot_res['OUT_MODEL'][stock]
                for itr in range(len(res1[ahead]['nit'])):
                    mse_value = res1[ahead][f'{metric.upper()}P'][itr]
                    mse_data[stock].append(mse_value)
    
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.boxplot([mse_data[stock] for stock in stocks], labels=stocks)
            ax.set_yscale('log')
            ax.set_title(f'{metric.upper()} - {selected_model.upper()}', fontsize=16)
            ax.set_xlabel('Acciones', fontsize=14)
            ax.set_ylabel(f'{metric.upper()}', fontsize=14)
            save_path = f'{plot_path}/{scen_name}/{tr_tst}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            fig_path = os.path.join(save_path, f'boxplot_{metric}_{selected_model}.{plot_format}')
            plt.savefig(fig_path)
            fig.show()

def plot_metric_boxplots_with_yesterday(selected_model: str, tr_tst_list: list, all_results: dict, stock_list: list, lahead: list, metric: str, scen_name: str, plot_path: str, plot_format: str) -> None:
    for stock in stock_list:
        fig, axs = plt.subplots(nrows=len(tr_tst_list), figsize=(8, 6 * len(tr_tst_list)))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        for idx, tr_tst in enumerate(tr_tst_list):
            metric_data = {ahead: [] for ahead in lahead}
            yesterday_data = {ahead: [] for ahead in lahead}
            pred_data = {ahead: [] for ahead in lahead}
            tot_res = all_results[tr_tst][selected_model]['tot_res']
            res1 = tot_res['OUT_MODEL'][stock]
            for ahead in lahead:
                for itr in range(len(res1[ahead]['nit'])):
                    metric_value = res1[ahead][f'{metric.upper()}P'][itr]
                    metric_data[ahead].append(metric_value)
                # Asegurarnos de que estamos tomando el último valor (última iteración)
                itr = len(res1[ahead]['nit']) - 1
                metricy = res1[ahead][f'{metric.upper()}Y'][itr]
                yesterday_data[ahead].append(metricy)
                metricp = res1[ahead][f'{metric.upper()}P'][itr]
                pred_data[ahead].append(metricp)

            axs[idx].boxplot([metric_data[ahead] for ahead in lahead], positions=range(len(lahead)))
            axs[idx].plot([yesterday_data[ahead] for ahead in lahead], 'r-o', label='Historical', linestyle='--')
            axs[idx].plot([pred_data[ahead] for ahead in lahead], 'g^', label='Predicted')

            #axs[idx].set_yscale('log')
            axs[idx].set_xticks(range(len(lahead)))
            axs[idx].set_xticklabels(lahead)
            axs[idx].set_title(f'{metric.upper()} - {selected_model.upper()} - {stock} - {tr_tst}', fontsize=16)
            axs[idx].set_ylabel(f'{metric}', fontsize=14)
            if idx == len(tr_tst_list) - 1:
                axs[idx].set_xlabel('Ahead days prediction (days)', fontsize=14)
            axs[idx].legend()

            plt.tight_layout()

            save_path = f'{plot_path}/{scen_name}/{metric.upper()}_{selected_model}_boxplot'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig_path = os.path.join(save_path, f'{stock}_boxplot_{metric}_{selected_model}.{plot_format}')
            plt.savefig(fig_path)
            fig.show()
