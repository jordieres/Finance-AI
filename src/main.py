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

def check_data_availability(config_file, required_data):
    config, _ = get_configuration(config_file)
    data_path = config['data']['data_path']
    output_data = config['data']['output_path']
    date = config['data']['date']
    

    if required_data == "pre":
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {data_path}")
            return False
    
    elif required_data == 'post':
        output_dir = os.path.join(output_data, "output")
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            print(f"The directory {output_dir} is missing or empty.")
            return False
    
    else:
        input_dir = os.path.join(output_data, "input")
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            print(f"The directory {input_dir} is missing or empty.")
            return False

    return True

def main(args):
    operations = args.operations.split(';')

    if "pre" in operations:
        if check_data_availability(args.params_file, required_data="pre"):
            run_dataprocessing_script(args.params_file)
        else:
            print("Required data for preprocessing is not available.")

    if "lstm" in operations:
        if check_data_availability(args.params_file, required_data="lstm"):
            run_lstm_script(args.params_file)
        else:
            print("Required data for LSTM is not available.")

    if "1DT" in operations:
        if check_data_availability(args.params_file, required_data="1DT"):
            run_unidimensional_transformer_script(args.params_file)
        else:
            print("Required data for UniDimensional Transformer is not available.")

    if "MDT" in operations:
        if check_data_availability(args.params_file, required_data="MDT"):
            run_multidimensional_transformer_script(args.params_file)
        else:
            print("Required data for MultiDimensional Transformer is not available.")

    if "post" in operations:
        if check_data_availability(args.params_file, required_data="post"):
            graphical_results(args.params_file)
        else:
            print("Required data for post-processing is not available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full system training, testing and results saving.")
    parser.add_argument("-c", "--params_file", required=True, nargs='?', action='store', help="Configuration file path")
    parser.add_argument("-o", "--operations", required=True, help="Operations to run: 'pre;lstm;1DT;MDT;post' you can run all or some or just one of them."
                        "pre: Data preprocessing, lstm: LSTM model training, 1DT: UniDimensional Transformer model training,"
                        "MDT: MultiDimensional Transformer model training, post: Results visualization.")
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Error: {e}\n")
        parser.print_help()
        sys.exit(2)
    except SystemExit as e:
        if e.code != 0:
            parser.print_help()
        sys.exit(e.code)
    main(args)
