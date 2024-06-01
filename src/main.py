import warnings
import os, sys
import subprocess
import random
import argparse

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from utils_vv_tfg import *
from config.config import get_configuration

def graphical_results(config_file):
    config, _ = get_configuration(config_file)
    output_path = config['data']['output_path']
    selected_scenario = config['visualization']['scenario']
    metric = config['visualization']['metric']
    plot_path = config['visualization']['plot_path']
    plot_format = config['visualization']['plot_format']

    for scen in config['scenarios']:
        if selected_scenario == scen['name']:
            stock_list = scen['tickers']
            list_tr_tst = scen['tr_tst']
            list_win_size = scen['win']
            lahead = scen['lahead']
            scen_name = scen['name']
    for win in list_win_size:
        all_results = {}
        for tr_tst in list_tr_tst:
            all_results[tr_tst] = load_output_preprocessed_data(output_path, win, tr_tst, selected_scenario)
        
        run_plot_res(list_tr_tst, all_results, stock_list, lahead, selected_scenario, metric, scen_name, plot_path, plot_format)

        for selected_model in all_results[tr_tst].keys():
            plot_metric_boxplots(selected_model, list_tr_tst, all_results, stock_list, lahead, metric, scen_name, plot_path, plot_format)
            plot_metric_boxplots_with_yesterday(selected_model, list_tr_tst, all_results, stock_list, lahead, metric, scen_name, plot_path, plot_format)

def run_dataprocessing_script(config_file):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_script_path = os.path.join(current_dir, "DataPreprocessing.py")

        subprocess.run(["python3", data_script_path, "-v", "1", "-c", config_file], check=True)
    except Exception as e:
        print("An error occurred while running DataPreprocessing.py:", e)

def run_lstm_script(config_file):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_script_path = os.path.join(current_dir, "ModellingLSTM.py")

        subprocess.run(["python3", lstm_script_path, "-c", config_file], check=True)
    except Exception as e:
        print("An error occurred while running ModellingLSTM.py:", e)

def run_unidimensional_transformer_script(config_file):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        transformer_script_path = os.path.join(current_dir, "UniDimTransformer.py")

        subprocess.run(["python3", transformer_script_path, "-c", config_file], check=True)
    except Exception as e:
        print("An error occurred while running Transformer.py:", e)

def run_multidimensional_transformer_script(config_file):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        multi_transformer_script_path = os.path.join(current_dir, "MultiDimTransformer.py")

        subprocess.run(["python3", multi_transformer_script_path, "-c", config_file], check=True)
    except Exception as e:
        print("An error occurred while running Transformer.py:", e)

def main(args):
    user_option = input("Do you want to run the complete system (y/n)? ")

    if user_option.lower() == "y":
        print("Running complete system")
        run_dataprocessing_script(args.params_file)
        run_lstm_script(args.params_file)
        run_unidimensional_transformer_script(args.params_file)
        run_multidimensional_transformer_script(args.params_file)
        graphical_results(args.params_file)
    elif user_option.lower() == "n":
        print("Running graphical results")
        graphical_results(args.params_file)
    else:
        print("Invalid option. Please try again.")
        main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM training, testing and results saving.")
    parser.add_argument("-c", "--params_file", nargs='?', action='store', help="Configuration file path")
    args = parser.parse_args()
    main(args)