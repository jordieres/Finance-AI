import warnings
import argparse
import yaml
import utils as ut

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


win_size = 5
stock = 'AAPL'
multi = False
list_tr_tst = [0.7, 0.8]
lahead = [1,7,14,30,90]
num_iterations = 10
for tr_tst in list_tr_tst:
    processed_path = f"D:/Escritorio/TFG/Finance-AI/DataProcessed/{tr_tst}"

    all_res = ut.load_output_preprocessed_data(processed_path, win_size, tr_tst, stock, multi)
    model_list = list(all_res.keys())
    model_results = all_res

    for itr in range(num_iterations):
        for ahead in lahead:
            ut.plot_results_comparison(model_results, model_list, stock, itr, ahead, tr_tst)
