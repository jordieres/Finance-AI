import yaml

def get_configuration():
    params_file = 'D:/Escritorio/TFG/Finance-AI/src/config/config.yaml'
    with open(params_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config, params_file