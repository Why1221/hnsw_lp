import json
import subprocess
from pathlib import Path

# Indexing for HNSW
datasets = ["glove"] #,gist","glove","deep","sift","sun","mnist"
p_lst = [1.0]  #[]
exp_path_org = None
M = 32
efc = 200

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_EXP_DIR = REPO_DIR.parent / "Experiments"
program_path = REPO_DIR / "hnsw_index"

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


def get_experiments_dir() -> Path:
    return Path(exp_path_org).expanduser().resolve() if exp_path_org else DEFAULT_EXP_DIR


def read_fvecs_shape(path: Path) -> tuple[int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as fin:
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


exp_path = get_experiments_dir()
if not program_path.exists():
    raise FileNotFoundError(f"hnsw_index not found: {program_path}")

for i in range(len(datasets)):
    for j in range(len(p_lst)):
        dataset = datasets[i]
        p = p_lst[j]
        dataset_dir = exp_path / dataset
        config_dir = dataset_dir / "config"
        train_file = dataset_dir / f"{dataset}-train.fvecs"
        query_file = dataset_dir / f"{dataset}-test.fvecs"

        if not train_file.exists():
            raise FileNotFoundError(f"Base data file not found: {train_file}")
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")
        
        # Try to read existing config, if fails use default
        ann_json_path = exp_path / "ann.json"
        try:
            with ann_json_path.open("r", encoding="utf-8") as fin:
                conf = json.load(fin)
        except (FileNotFoundError, json.JSONDecodeError):
            conf = default_config.copy()
            # Create ann.json if it doesn't exist
            ann_json_path.parent.mkdir(parents=True, exist_ok=True)
            with ann_json_path.open("w", encoding="utf-8") as fout:
                json.dump(default_config, fout, indent=4)
        
        # Update configuration with dataset-specific values
        n, d = read_fvecs_shape(train_file)
        conf["n"] = n
        conf["d"] = d
        conf["ds"] = str(train_file)
        conf["if"] = str(dataset_dir / f"{dataset}_{p}.index")
        conf["p"] = p
        conf["M"] = M
        conf["Ef"] = efc
        conf["nq"] = 1000
        conf["qf"] = str(query_file)
        conf["rf"] = str(dataset_dir / f"{dataset}_{p}_hnsw.res")
        conf["K"] = 100
        conf["efS"] = 1000

        # Create and save dataset-specific config
        config_file = config_dir / f"{dataset}_{p}.json"
        config_dir.mkdir(parents=True, exist_ok=True)
        with config_file.open("w", encoding="utf-8") as fout:
            json.dump(conf, fout, indent=4)
        
        print(f"Created config file: {config_file}")
        subprocess.run([str(program_path), str(config_file)], check=True)
