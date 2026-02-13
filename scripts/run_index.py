import subprocess
import pathlib
import os, sys
import json
import numpy as np

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

def read_fvecs_shape(path: str) -> tuple[int, int]:
    file_size = os.path.getsize(path)
    with open(path, "rb") as fin:
        dim_bytes = fin.read(4)
        if len(dim_bytes) != 4:
            raise RuntimeError(f"Failed to read dimension from fvecs file: {path}")
        dim = int.from_bytes(dim_bytes, byteorder="little", signed=False)
    if dim <= 0:
        raise RuntimeError(f"Invalid dimension in fvecs file: {path}")
    record_size = 4 + dim * 4
    if file_size % record_size != 0:
        raise RuntimeError(f"Corrupted fvecs file (record mismatch): {path}")
    num = file_size // record_size
    return int(num), int(dim)

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
        n,d =read_fvecs_shape(exp_path_org + "/{}/{}-train.fvecs".format(dataset,dataset))
        conf["n"] = n
        conf["d"] = d
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