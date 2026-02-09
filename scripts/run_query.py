import subprocess
import pathlib
import os, sys
import json
import numpy as np
from oniakIO import odats

# Indexing for HNSW
datasets = ["glove"] #"audio", "trevi", "gist","glove","deep",,"sun","mnist"
p_lst = [2.0]  #0.7,0.9,1.1,1.3,1.5,1.7,1.9
exp_path_org = "/media/gtnetuser/T7/Huayi/lp_graph/Experiments"
program_path = "/media/gtnetuser/T7/Huayi/lp_graph/hnsw_lp/hnsw_query"
M = 32
efc = 200
K = 200
efs = 1800

# Default configuration
default_config = {
    "M": M,
    "Ef": efc,
    "p": 2.0,
    "n": 0,  # will be updated per dataset
    "d": 0,  # will be updated per dataset
    "ds": "",  # will be updated per dataset
    "if": "" , # will be updated per dataset
    "nq": 1000,
    "qf": "",  # will be updated per dataset
    "rf": ""   # will be updated per dataset
}

for i in range(len(datasets)):
    for j in range(len(p_lst)):
        dataset = datasets[i]
        p = p_lst[j]
        
        # Try to read existing config, if fails use default
        ann_json_path = config_file = os.path.join(exp_path_org, "{}/config/{}_{}.json".format(dataset,dataset,p))
        try:
            with open(ann_json_path) as fin:
                conf = json.load(fin)
        except (FileNotFoundError, json.JSONDecodeError):
            conf = default_config.copy()
            # Create ann.json if it doesn't exist
            os.makedirs(os.path.dirname(ann_json_path), exist_ok=True)
            with open(ann_json_path, 'w') as fout:
                json.dump(default_config, fout, indent=4)
        
        # Update configuration with dataset-specific values
        query_data = odats.read_file(exp_path_org + "/{}/{}-test.fvecs".format(dataset,dataset))
        conf["nq"] = query_data.shape[0]
        conf["qf"] = exp_path_org + "/{}/{}-test.fvecs".format(dataset,dataset)
        conf["rf"] = exp_path_org + "/{}/{}_{}_{}.ivecs".format(dataset, dataset, p,K)
        conf["K"] = K
        conf["efS"] = efs

        # Create and save dataset-specific config
        config_file = os.path.join(exp_path_org, "{}/config/query_{}_{}_{}.json".format(dataset,dataset,p,K))
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as fout:
            json.dump(conf, fout, indent=4)
        
        print(f"Created config file: {config_file}")
        subprocess.run([program_path, config_file])