import json
import pandas as pd
import numpy as np
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

# input_file = 'C6KT68.jsonl'
# output_file = 'C6KT68.csv'

with open(input_file, 'r') as json_file:
    json_list = list(json_file)

fout = open(output_file,'w')
for json_str in json_list:
    tokens = json.loads(json_str)["features"]
    for token in tokens:
        if token['token'] in ['[CLS]','[SEP]']:
            continue
        else:
            last_layers = np.sum([
                token['layers'][0]['values'],
                token['layers'][1]['values'],
                token['layers'][2]['values'],
                token['layers'][3]['values'],
                token['layers'][4]['values'],
                token['layers'][5]['values'],
                token['layers'][6]['values'],
                token['layers'][7]['values'],
                token['layers'][8]['values'],
                token['layers'][9]['values'],
                token['layers'][10]['values'],
                token['layers'][11]['values'],
            ], axis=0)
            fout.write(f'{",".join(["{:f}".format(i) for i in last_layers])}\n')
    