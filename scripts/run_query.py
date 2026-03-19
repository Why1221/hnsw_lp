import json
import re
import subprocess
import sys
from pathlib import Path


# -----------------------------
# User parameters (edit here)
# -----------------------------
dataset = "sift"
p = 1.5
K = 50
efs = 400
t = 300  # used only when p is neither 1.0 nor 2.0

# Stage-2 recall parameter
tau = 0.92

# Paths
exp_path_org = None

# Fallback config defaults (used only if config json is missing/broken)
default_M = 32
default_Ef = 200

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_EXP_DIR = REPO_DIR.parent / "Experiments"
hnsw_query_path = REPO_DIR / "hnsw_query"
candidate_verify_path = REPO_DIR / "candidate_verify"


def format_p(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def is_close(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def parse_time_ms(output: str, pattern: str, stage_name: str) -> float:
    match = re.search(pattern, output)
    if match is None:
        raise RuntimeError(f"Failed to parse {stage_name} time from program output.")
    return float(match.group(1))


def run_command_show_output(cmd, name: str) -> str:
    print(f"Running {name}: {' '.join(str(part) for part in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}.")
    return result.stdout


def load_config_or_default(config_path: Path, default_config: dict) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as fin:
            return json.load(fin)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config.copy()


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


def get_experiments_dir() -> Path:
    return Path(exp_path_org).expanduser().resolve() if exp_path_org else DEFAULT_EXP_DIR


def main():
    if p <= 0:
        raise ValueError("p must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if efs <= 0:
        raise ValueError("efs must be > 0.")

    if not hnsw_query_path.exists():
        raise FileNotFoundError(f"hnsw_query not found: {hnsw_query_path}")

    direct_mode = is_close(p, 1.0) or is_close(p, 2.0)

    if not direct_mode:
        if t <= 10:
            raise ValueError("For two-stage mode, t must be > 10.")
        if t <= 4 * K:
            raise ValueError("For two-stage mode, t must be > 4K.")
        if not candidate_verify_path.exists():
            raise FileNotFoundError(f"candidate_verify not found: {candidate_verify_path}")

    exp_path = get_experiments_dir()
    dataset_dir = exp_path / dataset
    config_dir = dataset_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    data_file = dataset_dir / f"{dataset}-train.fvecs"
    query_file = dataset_dir / f"{dataset}-test.fvecs"
    if not data_file.exists():
        raise FileNotFoundError(f"Base data file not found: {data_file}")
    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")

    p_str = format_p(p)
    final_output_file = dataset_dir / f"{dataset}_{p_str}_{K}.ivecs"

    if direct_mode:
        stage1_p = p
        stage1_k = K
        stage1_output_file = final_output_file
    else:
        stage1_p = 2.0 if p > 1.4 else 1.0
        stage1_k = t
        stage1_output_file = dataset_dir / f"{dataset}_{p_str}_t{t}.ivecs"

    stage1_p_str = format_p(stage1_p)
    base_config_path = config_dir / f"{dataset}_{stage1_p_str}.json"

    default_config = {
        "M": default_M,
        "Ef": default_Ef,
        "p": stage1_p,
        "n": 0,
        "d": 0,
        "ds": str(data_file),
        "if": str(dataset_dir / f"{dataset}_{stage1_p_str}.index"),
        "nq": 0,
        "qf": str(query_file),
        "rf": str(stage1_output_file),
        "K": stage1_k,
        "efS": efs,
    }

    conf = load_config_or_default(base_config_path, default_config)

    nq, dim = read_fvecs_shape(query_file)
    conf["nq"] = nq
    conf["d"] = dim
    conf["ds"] = str(data_file)
    conf["if"] = str(dataset_dir / f"{dataset}_{stage1_p_str}.index")
    conf["qf"] = str(query_file)
    conf["rf"] = str(stage1_output_file)
    conf["K"] = int(stage1_k)
    conf["efS"] = int(efs)
    conf["p"] = float(stage1_p)
    conf["M"] = int(conf.get("M", default_M))
    conf["Ef"] = int(conf.get("Ef", default_Ef))

    if direct_mode:
        query_config_name = f"query_{dataset}_{p_str}_{K}.json"
    else:
        query_config_name = f"query_{dataset}_{p_str}_{K}_t{t}.json"
    query_config_path = config_dir / query_config_name

    index_file = dataset_dir / f"{dataset}_{stage1_p_str}.index"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    with query_config_path.open("w", encoding="utf-8") as fout:
        json.dump(conf, fout, indent=4)

    stage1_stdout = run_command_show_output([str(hnsw_query_path), str(query_config_path)], "hnsw_query")
    stage1_time_ms = parse_time_ms(stage1_stdout, r"Total time:\s*([0-9eE+\-.]+)\s*ms", "stage1")

    stage2_time_ms = 0.0
    if not direct_mode:
        verify_cmd = [
            str(candidate_verify_path),
            str(data_file),
            str(query_file),
            str(stage1_output_file),
            str(p),
            str(K),
            str(K),
            str(tau),
            str(final_output_file),
        ]
        stage2_stdout = run_command_show_output(verify_cmd, "candidate_verify")
        stage2_time_ms = parse_time_ms(
            stage2_stdout, r"Total Verification Time:\s*([0-9eE+\-.]+)\s*ms", "stage2"
        )

    if not final_output_file.exists():
        raise RuntimeError(f"Final output file was not generated: {final_output_file}")

    total_pipeline_time_ms = stage1_time_ms + stage2_time_ms
    print(f"Total Pipeline Time: {total_pipeline_time_ms:.4f} ms")


if __name__ == "__main__":
    main()
