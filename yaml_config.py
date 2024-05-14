import yaml

def getConfigYaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            return config_dict
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)

# add  configurations
def insertData(yaml_file,key, data):
    with open(yaml_file, encoding='utf-8') as file:
        dict_temp = yaml.load(file, Loader=yaml.FullLoader)
        try:
            dict_temp[key] = data
        except:
            if not dict_temp:
                dict_temp = {}
            dict_temp.update({key:data})
    with open(yaml_file,'w', encoding='utf-8') as file:
        yaml.dump(dict_temp, file, allow_unicode=True) # allow_unicode=True，解决存储时unicode编码问题

        
if __name__ == "__main__":
    a= getConfigYaml("parameters.yaml")
    sys_state = {}
    for item in a.items():
        sys_state[item[0]] = item[1]

    