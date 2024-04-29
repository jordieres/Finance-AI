import yaml

def get_configuration():
    params_file = '/home/vvallejo/Finance-AI/src/config/config.yaml'
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, params_file