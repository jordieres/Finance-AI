import warnings
import os, sys
import subprocess
import random

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
sys.path.append('/home/vvallejo/Finance-AI/src')

from utils_vv_tfg import load_output_preprocessed_data, plot_results_comparison, plot_ahead_perf_sameStock
from config.config import get_configuration

def graphical_results(select_scen):
    config, _ = get_configuration()
    processed_path = config['data']['output_path']

    for scenario in config['scenarios']:
        if scenario['name'] != select_scen:
            continue
        else:
            win_size = scenario['win']
            list_tr_tst = scenario['tr_tst']
            lahead = scenario['lahead']
            stock_list = scenario['tickers']
            epochs = scenario['epochs']
            bsize = scenario['batch_size']
            nhn = scenario['nhn']
            n_itr = scenario['n_itr']
            scen_name = scenario['name']
        
            for stock in stock_list:
                for tr_tst in list_tr_tst:
                    all_res_uni = load_output_preprocessed_data(win_size, tr_tst, False)
                    all_res_multi = load_output_preprocessed_data(win_size, tr_tst, True)
                    model_list_uni = list(all_res_uni.keys())
                    model_results_uni = all_res_uni
                    model_list_multi = list(all_res_multi.keys())
                    model_results_multi = all_res_multi
                    itr = random.randint(0, n_itr-1)
                    plot_results_comparison(model_results_uni, lahead, model_list_uni, scen_name, stock, itr, tr_tst)
                    plot_results_comparison(model_results_multi, lahead, model_list_multi, scen_name, stock, itr, tr_tst)
            plot_ahead_perf_sameStock(model_results_uni, lahead, model_list_uni, stock_list, list_tr_tst, scen_name)
            plot_ahead_perf_sameStock(model_results_multi, lahead, model_list_multi, stock_list, list_tr_tst, scen_name)

def run_dataprocessing_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_script_path = os.path.join(current_dir, "dataprocessed", "DataPreprocessing.py")

        subprocess.run(["python3", data_script_path, "-v", "1"], check=True)
    except Exception as e:
        print("An error occurred while running DataPreprocessing.py:", e)

def run_lstm_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_script_path = os.path.join(current_dir, "modelslstm", "ModellingLSTM.py")

        subprocess.run(["python3", lstm_script_path], check=True)
    except Exception as e:
        print("An error occurred while running ModellingLSTM.py:", e)

def run_unidimensional_transformer_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        transformer_script_path = os.path.join(current_dir, "modeltransformer", "UniDimTransformer.py")

        subprocess.run(["python3", transformer_script_path], check=True)
    except Exception as e:
        print("An error occurred while running Transformer.py:", e)

def run_multidimensional_transformer_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        multi_transformer_script_path = os.path.join(current_dir, "modeltransformer", "MultiDimTransformer.py")

        subprocess.run(["python3", multi_transformer_script_path], check=True)
    except Exception as e:
        print("An error occurred while running Transformer.py:", e)

def main():
    run_dataprocessing_script()
    #run_lstm_script()
    #run_unidimensional_transformer_script()
    #run_multidimensional_transformer_script()


if __name__ == "__main__":
    main()