# AI IN FINANCE: TRANSFORMERS APPLIED TO MULTIVARIATE TIME SERIES FORECASTING
Víctor Vallejo's final degree thesis (TFG) of the degree in Data Engineering and Systems at Universidad Politécnica de Madrid (UPM).

## Overview
This project aims to develop an artificial intelligence-based price prediction system for financial markets, leveraging transformative models in a multivariate environment. The architecture is designed to facilitate comparisons between transformer models and other AI technologies, applied to data from 15 different stocks within the S&P 500 index.

See the project's paper for more details:
```bibtex
@article{https://doi.org/10.5281/zenodo.11583436,
  doi = {10.5281/zenodo.11583436},
  url = {https://zenodo.org/doi/10.5281/zenodo.11493274},
  author = {Vallejo Carmona,  Víctor},
  language = {en},
  title = {AI in Finance: Transformers applied to multivariate time series forecasting},
  publisher = {Zenodo},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Features

- **Multivariate Data Analysis**: Incorporates data from various sources, including Twitter and Bloomberg sentiment analysis, daily trading volumes, and stock price-related factors.
- **Flexible Configuration**: Simulation scenarios can be customized via a YAML configuration file, allowing adaptability to various forecasting examples and contexts.
- **Multiple AI Models**: Includes different LSTM models (classic, stacked, and attention-based) developed using TensorFlow, and transformer models built with PyTorch.
- **Robust Visualization**: Presents model performance across different stocks and time horizons to evaluate effectiveness in various contexts.

## Data Collection and Preprocessing

- **Sources**: Data is collected from Twitter, Bloomberg, daily trading volumes, and stock price-related factors.
- **Storage**: Preprocessed data is stored in pickle files.
- **Preprocessing Steps**: 
  - Load raw data from CSV files.
  - Calculate sentiment scores.
  - Calculate volatility and trends.
  - Normalize the data for univariate and multivariate datasets.

## Simulation Scenarios

- Defined through a YAML file where users can specify parameters such as:
  - Stock tickers.
  - Train/test split ratios.
  - Window sizes.
  - Number of training iterations, epochs, and batch sizes.
  - Number of hidden neurons.
- Users can select between different LSTM models and customize transformer parameters.

## Model Architectures

### LSTM Models

1. **Basic LSTM**: Captures short-term dependencies.
2. **Stacked LSTM**: Captures more complex patterns.
3. **Attention-based LSTM**: Incorporates an attention mechanism for better prediction accuracy.

### Transformer Model

- Based on the classical architecture, including:
  - Embedding layers.
  - Encoder layers with multi-head attention.
  - Positional feed forward networks.

## Training and Evaluation

- **Optimizer**: AdamW
- **Loss Function**: MSELoss
- **Metrics**: 
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- Continuous evaluation to monitor performance and avoid overfitting.
- Predictions are denormalized to original scales.

## Usage

### Running the System

The operation of the system is managed by a flow of Python scripts executed by a main script. Users can choose to:
- Run the entire system.
- Display results from previous runs.

### Requirements

- Python 3.10.12
- TensorFlow
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jordieres/Finance-AI.git
   cd Finance-AI
   ```
2. Install the required packages:
   ```bash
   pip install -r requirement.txt
   ```

### Configuration
Customize the YAML configuration file to specify simulation parameters.

### Execution
Run the main script to start the simulation with the path of the configuration file and the operations to run:
operations: 'pre;lstm;1DT;MDT;post'
   ```bash
   python3 src/main.py -c path_to_config_file.yaml -o [operations]
   ```

## Results
Below are some example graphs of the results obtained from the models predictions:

### Amazon's stock price prediction using LSTM and transformer multivariate for the first of the scenarios.

![AMZN LSTM Price Prediction](figures/scenario_1/0.7/lstm-AMZN-9-MSE.png)
![AMZN Multivariate Transformer Price Prediction](figures/scenario_1/0.7/transformer-m-AMZN-9-MSE.png)

### Amazon's stock price prediction using LSTM and transformer multivariate for the first of the scenarios.

![AMZN LSTM Price Prediction](figures/scenario_1/0.7/lstm-AMZN-9-MSE.png)
![AMZN Multivariate Transformer Price Prediction](figures/scenario_1/0.7/transformer-m-AMZN-9-MSE.png)

### Apples's MSE comparison for each ahead value using LSTM and transformer multivariate for the first of the scenarios.

![AMZN LSTM Price Prediction](figures/scenario_1/MSE_lstm_boxplot/AAPL_boxplot_MSE_lstm.png)
![AMZN Multivariate Transformer Price Prediction](figures/scenario_1/MSE_transformer-m_boxplot/AAPL_boxplot_MSE_transformer-m.png)

## Application of trained Models
To load and apply the trained models to use the .h5 files for the LSTMs you can follow this example:

### With pre processed data:
#### Load input data:
```python
from utils_vv_tfg import load_preprocessed_data, denormalize_data
import numpy as np

lpar, tot_res = load_preprocessed_data(processed_path, win_size, tr_tst, stock, scenario_name, multi)
for ahead in lahead:
    tot = tot_res['INPUT_DATA'][ahead]
    testX  = tot['testX']
    vdd    = tot['vdd'] #data to denormalize the predictions
```

#### Load the Transformer pre trained model:
```python
import json
import numpy as np
import pandas as pd
import torch
from MultiDimTransformer import TransformerL as multiTransformer
from UniDimTransformer import TransformerL as uniTransformer
file_route = 'file_route'
json_file=open(f'{file_route}.json','r')
with open(f'{file_route}.json','r') as json_file:
    loaded_model_json = json.load(json_file)
vdd = loaded_model_json['vdd']
model = torch.load(f"{file_route}.h5")
```

### Without preprocessed data:

#### Load LSTM model trained:
```python
import json
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd

file_route = 'file_route'
json_file=open(f'{file_route}.json','r')
with open(f'{file_route}.json','r') as json_file:
    loaded_model_json = json.load(json_file)

vdd = loaded_model_json['vdd']
vdd = json.loads(vdd)
loaded_model=model_from_json(loaded_model_json['model'])
loaded_model.load_weights(f'{file_route}.h5')
print("Loaded 	model from disk")
```

#### Normalize the new data:
``` python
min_x, max_x, mean_x = vdd["min"], vdd["max"], vdd["mean"]
min_x = set(min_x.values())
max_x = set(max_x.values())
for m in min_x:
    min_x = m
for m in max_x:
    max_x = m
```

#### Make predictions with new data:
```python
X = np.array(new_data)
X = pd.DataFrame(X)
x_mean = np.mean(X, axis=1)
center_x = X.sub(x_mean, axis=0)
center_x_norm = center_x.apply(lambda x: (x - min_x) / (max_x - min_x), axis=1)
loaded_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
score=loaded_model.predict(center_x_norm)
```


## Contributing
Contributions are welcome!

## Acknowledgments:

Special thanks to the ETSIST Department of Ingeniería de Organización, Administración de Empresas y Estadística for mentoring this project and for providing the data for this study.


