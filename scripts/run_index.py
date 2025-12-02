import subprocess
import pathlib
import os, sys
import json
import numpy as np
from oniakIO import odats

# Indexing for HNSW
datasets = ["glove"] #,gist","glove","deep","sift","sun","mnist"
p_lst = [1.0]  #[]
exp_path_org = "/media/gtnetuser/T7/Huayi/lp_graph/Experiments"
program_path = "/media/gtnetuser/T7/Huayi/lp_graph/hnsw_lp/hnsw_index"
M = 32
efc = 200

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
        ann_json_path = os.path.join(exp_path_org, "ann.json")
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
        data = odats.read_file(exp_path_org + "/{}/{}-train.fvecs".format(dataset,dataset))
        conf["n"] = data.shape[0]
        conf["d"] = data.shape[1]
        conf["ds"] = exp_path_org + "/{}/{}-train.fvecs".format(dataset,dataset)
        conf["if"] = exp_path_org + "/{}/{}_{}.index".format(dataset, dataset, p)
        conf["p"] = p
        conf["M"] = M
        conf["Ef"] = efc
        conf["nq"] = 1000
        conf["qf"] = exp_path_org + "/{}/{}-test.fvecs".format(dataset,dataset)
        conf["rf"] = exp_path_org + "/{}/{}_{}_hnsw.res".format(dataset, dataset, p)
        conf["K"] = 100
        conf["efS"] = 1000

        # Create and save dataset-specific config
        config_file = os.path.join(exp_path_org, "{}/config/{}_{}.json".format(dataset,dataset,p))
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as fout:
            json.dump(conf, fout, indent=4)
        
        print(f"Created config file: {config_file}")
        subprocess.run([program_path, config_file])