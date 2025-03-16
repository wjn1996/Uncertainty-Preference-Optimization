import yaml
 
def read_yaml_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data