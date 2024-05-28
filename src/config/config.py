import yaml

def get_configuration(params_file: str):
    try:
        with open(params_file, 'r') as f:
            config = yaml.safe_load(f)
        return config, params_file
    # Handle file not found error
    except FileNotFoundError:
        print(f"Error: The file {params_file} does not exist.")
        return None, params_file
    # Handle YAML parsing error
    except yaml.YAMLError as e:
        print(f"Error while reading YAML file: {e}")
        return None, params_file
    # Handle any other unexpected errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, params_file