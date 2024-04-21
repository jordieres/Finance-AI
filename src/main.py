import warnings
import os, sys
import subprocess

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
sys.path.append('D:/Escritorio/TFG/Finance-AI/src')

from utils_vv_tfg import load_output_preprocessed_data, plot_results_comparison
from config.config import get_configuration

def graphical_results():
    config, _ = get_configuration()
    processed_path = config['data']['output_path']
    multi = config['multi']
    scenarios = []
    for scenario in config['scenarios']:
        win_size = scenario['win']
        list_tr_tst = scenario['tr_tst']
        lahead = scenario['lahead']
        stock_list = scenario['tickers']
        epochs = scenario['epochs']
        bsize = scenario['batch_size']
        nhn = scenario['nhn']
        n_itr = scenario['n_itr']
        scen_name = scenario['name']
    # for stock in stock_list:
    for stock in ['AAPL']:
        for tr_tst in list_tr_tst:
            all_res = load_output_preprocessed_data(win_size, tr_tst, multi)
            model_list = list(all_res.keys())
            model_results = all_res

            for itr in range(n_itr):
                for ahead in lahead:
                    plot_results_comparison(model_results, model_list, scen_name, stock, itr, ahead, tr_tst)

def run_dataprocessing_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_script_path = os.path.join(current_dir, "DataPreprocessing", "DataPreprocessing.py")

        subprocess.run(["python", data_script_path, "-v", "1"])
    except Exception as e:
        print("Ocurrió un error al ejecutar DataPreprocessing.py:", e)

def run_lstm_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_script_path = os.path.join(current_dir, "ModelsLSTM", "ModellingLSTM.py")

        subprocess.run(["python", lstm_script_path])
    except Exception as e:
        print("Ocurrió un error al ejecutar ModellingLSTM.py:", e)

def run_transformer_script():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        transformer_script_path = os.path.join(current_dir, "ModelTransformer", "Transformer.py")

        subprocess.run(["python", transformer_script_path])
    except Exception as e:
        print("Ocurrió un error al ejecutar Transformer.py:", e)



if __name__ == "__main__":
    run_dataprocessing_script()
    run_lstm_script()
    run_transformer_script()
    graphical_results()