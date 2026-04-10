#!/usr/bin/env bash
#  PYTHON_BIN=/home/yaoyi/coding/dev/aic/collect_data/aiconfigurator/myenv/bin/python bash run_cli_default_matrix_agg_in_container.sh cli_default_matrix_results.csv --container vllm-xpu-0_17_0
#  PYTHON_BIN=/home/yaoyi/coding/dev/aic/collect_data/aiconfigurator/myenv/bin/python bash run_cli_default_matrix_agg_in_container.sh cli_default_matrix_results.csv --container vllm-xpu-0_17_0 --max-num-batched-tokens-mode auto
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICTION_CSV="${PREDICTION_CSV:-$ROOT_DIR/cli_default_matrix_results.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/cli_default_matrix_agg_measured_results.csv}"
RAW_LOG_DIR="${RAW_LOG_DIR:-$ROOT_DIR/cli_default_matrix_container_logs}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
PORT="${PORT:-8100}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
STARTUP_POLL_INTERVAL="${STARTUP_POLL_INTERVAL:-5}"
MAX_NUM_BATCHED_TOKENS_MODE="${MAX_NUM_BATCHED_TOKENS_MODE:-fixed}"
MAX_NUM_BATCHED_TOKENS_VALUE="${MAX_NUM_BATCHED_TOKENS_VALUE:-2048}"
HOST_NAME="${HOST_NAME:-localhost}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
AIPERF_VENV="${AIPERF_VENV:-/workspace/aiperf/bin/activate}"

FILTER_MODEL=""
FILTER_GPUS=""
FILTER_ISL=""
FILTER_OSL=""
FILTER_TTFT_INPUT=""
FILTER_TPOT_INPUT=""
POSITIONAL_PREDICTION_CSV=""

usage() {
  cat <<'EOF'
Usage:
  bash run_cli_default_matrix_agg_in_container.sh --container <name-or-id> [prediction_csv] [options]

Required:
  --container <name-or-id>          Container to exec into

Optional filters:
  --model <model>
  --gpus <count>
  --isl <tokens>
  --osl <tokens>
  --ttft-input <ms>
  --tpot-input <ms>

Optional runtime settings:
  --prediction-csv <path>
  --output-csv <path>
  --raw-log-dir <path>
  --engine <docker|podman>
  --port <port>
  --startup-timeout <seconds>
  --startup-poll-interval <seconds>
  --max-num-batched-tokens-mode <fixed|auto>
  --max-num-batched-tokens-value <value>
  --aiperf-venv <path>

Environment overrides are also supported for all of the settings above.
If prediction_csv is omitted, the default is ./cli_default_matrix_results.csv.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --prediction-csv)
      PREDICTION_CSV="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --raw-log-dir)
      RAW_LOG_DIR="$2"
      shift 2
      ;;
    --engine)
      CONTAINER_ENGINE="$2"
      shift 2
      ;;
    --model)
      FILTER_MODEL="$2"
      shift 2
      ;;
    --gpus)
      FILTER_GPUS="$2"
      shift 2
      ;;
    --isl)
      FILTER_ISL="$2"
      shift 2
      ;;
    --osl)
      FILTER_OSL="$2"
      shift 2
      ;;
    --ttft-input)
      FILTER_TTFT_INPUT="$2"
      shift 2
      ;;
    --tpot-input)
      FILTER_TPOT_INPUT="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --startup-poll-interval)
      STARTUP_POLL_INTERVAL="$2"
      shift 2
      ;;
    --max-num-batched-tokens-mode)
      MAX_NUM_BATCHED_TOKENS_MODE="$2"
      shift 2
      ;;
    --max-num-batched-tokens-value)
      MAX_NUM_BATCHED_TOKENS_VALUE="$2"
      shift 2
      ;;
    --aiperf-venv)
      AIPERF_VENV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$POSITIONAL_PREDICTION_CSV" && "$1" != --* ]]; then
        POSITIONAL_PREDICTION_CSV="$1"
        shift
      else
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -n "$POSITIONAL_PREDICTION_CSV" ]]; then
  PREDICTION_CSV="$POSITIONAL_PREDICTION_CSV"
fi

if [[ -z "$CONTAINER_NAME" ]]; then
  echo "--container is required" >&2
  usage >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/myenv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/myenv/bin/python"
else
  PYTHON_CMD="python"
fi

mkdir -p "$RAW_LOG_DIR"

export PREDICTION_CSV OUTPUT_CSV RAW_LOG_DIR CONTAINER_ENGINE CONTAINER_NAME PORT
export STARTUP_TIMEOUT STARTUP_POLL_INTERVAL MAX_NUM_BATCHED_TOKENS_MODE MAX_NUM_BATCHED_TOKENS_VALUE
export HOST_NAME GPU_MEMORY_UTILIZATION MAX_MODEL_LEN BLOCK_SIZE AIPERF_VENV
export FILTER_MODEL FILTER_GPUS FILTER_ISL FILTER_OSL FILTER_TTFT_INPUT FILTER_TPOT_INPUT

"$PYTHON_CMD" - <<'PY'
import csv
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


prediction_csv = Path(os.environ["PREDICTION_CSV"])
output_csv = Path(os.environ["OUTPUT_CSV"])
raw_log_dir = Path(os.environ["RAW_LOG_DIR"])
container_engine = os.environ["CONTAINER_ENGINE"]
container_name = os.environ["CONTAINER_NAME"]
port = int(os.environ["PORT"])
startup_timeout = int(os.environ["STARTUP_TIMEOUT"])
startup_poll_interval = int(os.environ["STARTUP_POLL_INTERVAL"])
max_num_batched_tokens_mode = os.environ["MAX_NUM_BATCHED_TOKENS_MODE"].strip().lower()
max_num_batched_tokens_value = int(os.environ["MAX_NUM_BATCHED_TOKENS_VALUE"])
host_name = os.environ["HOST_NAME"]
gpu_memory_utilization = os.environ["GPU_MEMORY_UTILIZATION"]
max_model_len = os.environ["MAX_MODEL_LEN"]
block_size = os.environ["BLOCK_SIZE"]
aiperf_venv = os.environ["AIPERF_VENV"]

filter_model = os.environ["FILTER_MODEL"].strip()
filter_gpus = os.environ["FILTER_GPUS"].strip()
filter_isl = os.environ["FILTER_ISL"].strip()
filter_osl = os.environ["FILTER_OSL"].strip()
filter_ttft_input = os.environ["FILTER_TTFT_INPUT"].strip()
filter_tpot_input = os.environ["FILTER_TPOT_INPUT"].strip()

ansi_escape_re = re.compile(r"\x1b\[[0-9;]*m")
leading_int_re = re.compile(r"^\s*(\d+)")
parallel_tp_re = re.compile(r"tp(\d+)")

metric_patterns = {
  "measured_TTFT (ms)": [
    re.compile(r"TTFT\s*\(ms\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bTTFT\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE),
  ],
  "measured_TPOT (ms)": [
    re.compile(r"TPOT\s*\(ms\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bTPOT\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE),
  ],
  "measured_tokens/s/gpu": [
    re.compile(r"tokens/s/gpu\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
  ],
  "measured_tokens/s/user": [
    re.compile(r"tokens/s/user\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
  ],
  "measured_req/s": [
    re.compile(r"\breq/s\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"Request Rate\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*req/s", re.IGNORECASE),
  ],
}


def strip_ansi(text: str) -> str:
  return ansi_escape_re.sub("", text)


def normalize_int(value: str) -> int | None:
  if not value:
    return None
  match = leading_int_re.match(value)
  if not match:
    return None
  return int(match.group(1))


def parse_tp(parallel: str) -> int:
  match = parallel_tp_re.search(parallel or "")
  if not match:
    raise ValueError(f"Unable to parse tp from parallel={parallel!r}")
  return int(match.group(1))


def affinity_mask(num_gpus: int) -> str:
  return ",".join(str(index) for index in range(num_gpus))


def sanitize_name(value: str) -> str:
  sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
  return sanitized.strip("_") or "case"


def derive_tokenizer_name(model: str) -> str:
  model_path = Path(model)
  for part in model_path.parts:
    if not part.startswith("models--"):
      continue
    repo = part[len("models--") :].replace("--", "/", 1)
    if "/" in repo:
      return repo
  return model


def derive_request_model_name(model: str) -> str:
  model_path = Path(model)
  if model_path.is_absolute() and "snapshots" in model_path.parts:
    return derive_tokenizer_name(model)
  return model


def derive_vllm_model_name(model: str) -> str:
  model_path = Path(model)
  if model_path.is_absolute() and "snapshots" in model_path.parts:
    return derive_tokenizer_name(model)
  return model


def normalize_metric_value(value: str) -> str:
  return value.replace(",", "").strip()


def parse_float(value: str) -> float | None:
  try:
    return float(normalize_metric_value(value))
  except (TypeError, ValueError):
    return None


def pct_delta(predicted: float, measured: float) -> float | None:
  if predicted == 0:
    return None
  return (measured - predicted) / predicted * 100.0


def add_comparison_metrics(output_row: dict[str, str]) -> None:
  metric_pairs = [
    ("TTFT (ms)", "predicted_TTFT (ms)", "measured_TTFT (ms)"),
    ("TPOT (ms)", "predicted_TPOT (ms)", "measured_TPOT (ms)"),
    ("tokens/s/gpu", "predicted_tokens/s/gpu", "measured_tokens/s/gpu"),
    ("tokens/s/user", "predicted_tokens/s/user", "measured_tokens/s/user"),
    ("req/s", "predicted_req/s", "measured_req/s"),
  ]
  for label, predicted_key, measured_key in metric_pairs:
    predicted = parse_float(output_row.get(predicted_key, ""))
    measured = parse_float(output_row.get(measured_key, ""))
    delta_key = f"delta_{label}"
    pct_key = f"delta_pct_{label}"
    output_row[delta_key] = ""
    output_row[pct_key] = ""
    if predicted is None or measured is None:
      continue
    output_row[delta_key] = f"{measured - predicted:.2f}"
    pct = pct_delta(predicted, measured)
    if pct is not None:
      output_row[pct_key] = f"{pct:.2f}"


def log(message: str = "") -> None:
  print(message, flush=True)


def log_separator(char: str = "=", width: int = 72) -> None:
  log(char * width)


def log_section(title: str) -> None:
  log()
  log_separator("=")
  log(title)
  log_separator("=")


def log_subsection(title: str) -> None:
  log()
  log(f"[{title}]")


def log_kv(key: str, value: str) -> None:
  log(f"  {key}: {value}")


def log_command(title: str, command: str) -> None:
  log_subsection(title)
  log(command)


def log_measured_results(output_row: dict[str, str]) -> None:
  log_subsection("Measured Results")
  log_kv("TTFT (ms)", output_row.get("measured_TTFT (ms)", "") or "N/A")
  log_kv("TPOT (ms)", output_row.get("measured_TPOT (ms)", "") or "N/A")
  log_kv("tokens/s/gpu", output_row.get("measured_tokens/s/gpu", "") or "N/A")
  log_kv("tokens/s/user", output_row.get("measured_tokens/s/user", "") or "N/A")
  log_kv("req/s", output_row.get("measured_req/s", "") or "N/A")


def log_benchmark_failure(returncode: int, benchmark_output: str) -> None:
  log_subsection("AIPerf Failure")
  log_kv("exit code", str(returncode))
  if benchmark_output.strip():
    log("AIPerf output:")
    log(benchmark_output.rstrip())


def run_host_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
  result = subprocess.run(cmd, capture_output=True, text=True)
  if check and result.returncode != 0:
    raise RuntimeError(
      f"Host command failed ({result.returncode}): {shlex.join(cmd)}\n"
      f"stdout:\n{result.stdout}\n"
      f"stderr:\n{result.stderr}"
    )
  return result


def run_container_script(script: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  cmd = [container_engine, "exec", container_name, "bash", "-lc", script]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if check and result.returncode != 0:
    raise RuntimeError(
      f"Container command failed ({result.returncode}): {shlex.join(cmd)}\n"
      f"stdout:\n{result.stdout}\n"
      f"stderr:\n{result.stderr}"
    )
  return result


def fetch_container_file(path: str) -> str:
  result = run_container_script(
    f"set +e; if [[ -f {shlex.quote(path)} ]]; then cat {shlex.quote(path)}; fi",
    check=False,
  )
  return result.stdout


def ensure_container_exists() -> None:
  run_host_command([container_engine, "inspect", container_name])


def list_vllm_processes() -> list[tuple[int, str]]:
  script = r'''
python3 - <<'__AIC_LIST_VLLM__'
import subprocess

result = subprocess.run(
  ["ps", "-eo", "pid=,args="],
  capture_output=True,
  text=True,
  check=True,
)

for raw_line in result.stdout.splitlines():
  line = raw_line.strip()
  if not line:
    continue
  parts = line.split(None, 1)
  if len(parts) != 2:
    continue
  pid_text, args = parts
  if "vllm serve" not in args and "vllm.entrypoints.openai.api_server" not in args:
    continue
  if "__AIC_LIST_VLLM__" in args or "ps -eo pid=,args=" in args:
    continue
  print(f"{pid_text}\t{args}")
__AIC_LIST_VLLM__
'''
  result = run_container_script(script, check=False)
  processes: list[tuple[int, str]] = []
  for line in result.stdout.splitlines():
    if not line.strip():
      continue
    pid_text, _, args = line.partition("\t")
    try:
      processes.append((int(pid_text.strip()), args.strip()))
    except ValueError:
      continue
  return processes


def wait_for_process_exit(timeout_seconds: int) -> None:
  deadline = time.time() + timeout_seconds
  while time.time() < deadline:
    if not list_vllm_processes():
      return
    time.sleep(1)
  raise RuntimeError("Timed out waiting for existing vLLM processes to exit")


def kill_existing_vllm(port_value: int) -> str:
  processes = list_vllm_processes()
  output_lines: list[str] = []
  if processes:
    output_lines.extend(f"{pid}\t{args}" for pid, args in processes)
    pid_list = " ".join(str(pid) for pid, _ in processes)
    run_container_script(f"set +e; kill {pid_list}", check=False)
    try:
      wait_for_process_exit(timeout_seconds=10)
    except RuntimeError:
      remaining = list_vllm_processes()
      if remaining:
        pid_list = " ".join(str(pid) for pid, _ in remaining)
        run_container_script(f"set +e; kill -9 {pid_list}", check=False)
        wait_for_process_exit(timeout_seconds=10)
  else:
    output_lines.append("No existing vLLM service process found.")

  run_container_script(
    f"set +e; if command -v fuser >/dev/null 2>&1; then fuser -k {port_value}/tcp >/dev/null 2>&1 || true; fi; rm -f /tmp/aic_vllm.pid",
    check=False,
  )
  return "\n".join(output_lines)


def wait_for_vllm_ready(port_value: int, timeout_seconds: int, expected_model_name: str) -> None:
  deadline = time.time() + timeout_seconds
  health_script = f"""
python3 - <<'__AIC_HEALTHCHECK__'
import sys
import urllib.request
import json

url = 'http://localhost:{port_value}/v1/models'
try:
  with urllib.request.urlopen(url, timeout=5) as response:
    payload = json.loads(response.read().decode())
    model_ids = [item.get('id', '') for item in payload.get('data', [])]
    if 200 <= response.status < 300 and {expected_model_name!r} in model_ids:
      print(response.status)
      sys.exit(0)
    print(response.status, model_ids)
    sys.exit(1)
except Exception as exc:
  print(exc)
  sys.exit(1)
__AIC_HEALTHCHECK__
"""
  alive_script = """
set +e
if [[ -f /tmp/aic_vllm.pid ]]; then
  pid=$(cat /tmp/aic_vllm.pid)
  kill -0 "$pid" >/dev/null 2>&1
  exit $?
fi
exit 1
"""
  while time.time() < deadline:
    health_result = run_container_script(health_script, check=False)
    if health_result.returncode == 0:
      return
    alive_result = run_container_script(alive_script, check=False)
    if alive_result.returncode != 0:
      raise RuntimeError("vLLM process exited before becoming ready")
    time.sleep(startup_poll_interval)
  raise RuntimeError(f"Timed out waiting for vLLM readiness on port {port_value}")


def start_vllm(
  model: str,
  served_model_name: str,
  tp: int,
  num_gpus: int,
  max_num_batched_tokens: int,
  log_path: str,
) -> str:
  display_command = f"""
export HF_HUB_OFFLINE=1
export ZE_AFFINITY_MASK={shlex.quote(affinity_mask(num_gpus))}
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
nohup vllm serve {shlex.quote(model)} \
  -tp {tp} \
  --served-model-name {shlex.quote(served_model_name)} \
  --host {shlex.quote(host_name)} \
  --port {port} \
  --seed 42 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization {shlex.quote(gpu_memory_utilization)} \
  --max-model-len {shlex.quote(max_model_len)} \
  --max_num_batched_tokens {max_num_batched_tokens} \
  --block-size {shlex.quote(block_size)} \
  --no-enable-prefix-caching \
  > {shlex.quote(log_path)} 2>&1 &
""".strip()
  log("vLLM launch command:")
  log(display_command)
  start_script = f"""
set -euo pipefail
mkdir -p $(dirname {shlex.quote(log_path)})
: > {shlex.quote(log_path)}
export HF_HUB_OFFLINE=1
export ZE_AFFINITY_MASK={shlex.quote(affinity_mask(num_gpus))}
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
nohup vllm serve {shlex.quote(model)} \
  -tp {tp} \
  --served-model-name {shlex.quote(served_model_name)} \
  --host {shlex.quote(host_name)} \
  --port {port} \
  --seed 42 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization {shlex.quote(gpu_memory_utilization)} \
  --max-model-len {shlex.quote(max_model_len)} \
  --max_num_batched_tokens {max_num_batched_tokens} \
  --block-size {shlex.quote(block_size)} \
  --no-enable-prefix-caching \
  > {shlex.quote(log_path)} 2>&1 &
echo $! > /tmp/aic_vllm.pid
cat /tmp/aic_vllm.pid
"""
  result = run_container_script(start_script)
  return result.stdout.strip()


def parse_metrics(output_text: str, num_gpus: int, osl: int) -> dict[str, str]:
  cleaned = strip_ansi(output_text)
  parsed: dict[str, str] = {}
  for key, patterns in metric_patterns.items():
    for pattern in patterns:
      matches = pattern.findall(cleaned)
      if matches:
        parsed[key] = normalize_metric_value(matches[-1])
        break
    else:
      parsed[key] = ""

  table_patterns = {
    "measured_TTFT (ms)": re.compile(r"Time to First Token\s*│\s*([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE),
    "measured_TPOT (ms)": re.compile(r"Inter Token Latency\s*│\s*([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE),
    "_output_token_throughput": re.compile(r"Output Token Throughput\s*│\s*([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE),
    "measured_req/s": re.compile(r"Request Throughput\s*│\s*([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE),
  }

  for key, pattern in table_patterns.items():
    if key in parsed and parsed.get(key):
      continue
    match = pattern.search(cleaned)
    if match:
      parsed[key] = normalize_metric_value(match.group(1))

  if not parsed.get("measured_tokens/s/user") and parsed.get("measured_TPOT (ms)"):
    tpot_ms = float(parsed["measured_TPOT (ms)"])
    if tpot_ms > 0:
      parsed["measured_tokens/s/user"] = f"{1000.0 / tpot_ms:.2f}"

  output_token_throughput = parsed.get("_output_token_throughput", "")
  if not parsed.get("measured_tokens/s/gpu"):
    if output_token_throughput and num_gpus > 0:
      parsed["measured_tokens/s/gpu"] = f"{float(output_token_throughput) / num_gpus:.2f}"
    elif parsed.get("measured_req/s") and num_gpus > 0:
      parsed["measured_tokens/s/gpu"] = f"{float(parsed['measured_req/s']) * osl / num_gpus:.2f}"

  parsed.pop("_output_token_throughput", None)
  return parsed


def parse_metrics_from_export(csv_text: str, num_gpus: int) -> dict[str, str]:
  parsed = {
    "measured_TTFT (ms)": "",
    "measured_TPOT (ms)": "",
    "measured_tokens/s/gpu": "",
    "measured_tokens/s/user": "",
    "measured_req/s": "",
  }
  if not csv_text.strip():
    return parsed

  rows = list(csv.DictReader(csv_text.splitlines()))
  metric_rows = {row.get("Metric", ""): row for row in rows}

  def get_avg(prefix: str) -> float | None:
    for name, row in metric_rows.items():
      if name.startswith(prefix):
        return parse_float(row.get("avg", ""))
    return None

  ttft = get_avg("Time to First Token")
  tpot = get_avg("Inter Token Latency")
  throughput_gpu_total = get_avg("Output Token Throughput (tokens/sec)")
  throughput_user = get_avg("Output Token Throughput Per User")
  req_s = get_avg("Request Throughput (requests/sec)")

  if ttft is not None:
    parsed["measured_TTFT (ms)"] = f"{ttft:.2f}"
  if tpot is not None:
    parsed["measured_TPOT (ms)"] = f"{tpot:.2f}"
  if throughput_user is not None:
    parsed["measured_tokens/s/user"] = f"{throughput_user:.2f}"
  if req_s is not None:
    parsed["measured_req/s"] = f"{req_s:.2f}"
  if throughput_gpu_total is not None and num_gpus > 0:
    parsed["measured_tokens/s/gpu"] = f"{throughput_gpu_total / num_gpus:.2f}"

  return parsed


def case_matches(row: dict[str, str]) -> bool:
  if row.get("P/D") != "agg":
    return False
  if filter_model and row.get("Model") != filter_model:
    return False
  if filter_gpus and row.get("# of XPUs") != filter_gpus:
    return False
  if filter_isl and row.get("ISL") != filter_isl:
    return False
  if filter_osl and row.get("OSL") != filter_osl:
    return False
  if filter_ttft_input and row.get("TTFT_input") != filter_ttft_input:
    return False
  if filter_tpot_input and row.get("TPOT_input") != filter_tpot_input:
    return False
  return True


def make_output_row(row: dict[str, str]) -> dict[str, str]:
  return {
    "ISL": row.get("ISL", ""),
    "OSL": row.get("OSL", ""),
    "TTFT_input": row.get("TTFT_input", ""),
    "TPOT_input": row.get("TPOT_input", ""),
    "Model": row.get("Model", ""),
    "# of XPUs": row.get("# of XPUs", ""),
    "Parallel": row.get("Parallel", ""),
    "max_num_batched_tokens": "",
    "Predicted BS": row.get("BS", ""),
    "Predicted concurrency": row.get("concurrency", ""),
    "predicted_TTFT (ms)": row.get("TTFT (ms)", ""),
    "predicted_TPOT (ms)": row.get("TPOT (ms)", ""),
    "predicted_tokens/s/gpu": row.get("tokens/s/gpu", ""),
    "predicted_tokens/s/user": row.get("tokens/s/user", ""),
    "predicted_req/s": row.get("req/s", ""),
    "measured_TTFT (ms)": "",
    "measured_TPOT (ms)": "",
    "measured_tokens/s/gpu": "",
    "measured_tokens/s/user": "",
    "measured_req/s": "",
    "delta_TTFT (ms)": "",
    "delta_pct_TTFT (ms)": "",
    "delta_TPOT (ms)": "",
    "delta_pct_TPOT (ms)": "",
    "delta_tokens/s/gpu": "",
    "delta_pct_tokens/s/gpu": "",
    "delta_tokens/s/user": "",
    "delta_pct_tokens/s/user": "",
    "delta_req/s": "",
    "delta_pct_req/s": "",
    "status": "",
    "notes": "",
  }


def build_derivation_key(row: dict[str, str], max_num_batched_tokens: int) -> tuple[str, ...]:
  return (
    row.get("Model", ""),
    row.get("ISL", ""),
    row.get("OSL", ""),
    row.get("TTFT_input", ""),
    row.get("TPOT_input", ""),
    row.get("Parallel", ""),
    row.get("BS", ""),
    str(max_num_batched_tokens),
  )


def build_scenario_key(row: dict[str, str], max_num_batched_tokens: int) -> tuple[str, ...]:
  return (
    row.get("Model", ""),
    row.get("ISL", ""),
    row.get("OSL", ""),
    row.get("TTFT_input", ""),
    row.get("TPOT_input", ""),
    str(max_num_batched_tokens),
  )


def has_complete_measured_metrics(row: dict[str, str]) -> bool:
  required_keys = [
    "measured_TTFT (ms)",
    "measured_TPOT (ms)",
    "measured_tokens/s/gpu",
    "measured_tokens/s/user",
    "measured_req/s",
  ]
  return all(row.get(key, "") for key in required_keys)


def find_derivation_source(
  derived_rows_by_key: dict[tuple[str, ...], list[dict[str, object]]],
  key: tuple[str, ...],
  target_num_gpus: int,
  target_concurrency: int,
) -> dict[str, object] | None:
  candidates = []
  for entry in derived_rows_by_key.get(key, []):
    source_num_gpus = entry["num_gpus"]
    source_concurrency = entry["concurrency"]
    source_row = entry["row"]
    if not isinstance(source_num_gpus, int) or not isinstance(source_concurrency, int):
      continue
    if target_num_gpus <= source_num_gpus:
      continue
    if target_num_gpus % source_num_gpus != 0:
      continue
    if source_concurrency * (target_num_gpus // source_num_gpus) != target_concurrency:
      continue
    if not isinstance(source_row, dict) or not has_complete_measured_metrics(source_row):
      continue
    candidates.append(entry)

  if not candidates:
    return None

  candidates.sort(key=lambda item: int(item["num_gpus"]))
  return candidates[0]


def has_same_scenario_smaller_gpu_case(
  scenario_rows_by_key: dict[tuple[str, ...], list[dict[str, object]]],
  key: tuple[str, ...],
  target_num_gpus: int,
) -> bool:
  for entry in scenario_rows_by_key.get(key, []):
    source_num_gpus = entry.get("num_gpus")
    if isinstance(source_num_gpus, int) and source_num_gpus < target_num_gpus:
      return True
  return False


def derive_output_row_from_source(
  output_row: dict[str, str],
  source_row: dict[str, str],
  scale_factor: int,
  source_num_gpus: int,
) -> None:
  output_row["measured_TTFT (ms)"] = source_row["measured_TTFT (ms)"]
  output_row["measured_TPOT (ms)"] = source_row["measured_TPOT (ms)"]

  throughput_keys = [
    "measured_tokens/s/gpu",
    "measured_tokens/s/user",
    "measured_req/s",
  ]
  for key in throughput_keys:
    value = parse_float(source_row.get(key, ""))
    if value is None:
      output_row[key] = ""
      continue
    output_row[key] = f"{value * scale_factor:.2f}"

  add_comparison_metrics(output_row)
  output_row["status"] = "derived"
  output_row["notes"] = (
    f"Derived from {source_num_gpus}-GPU measured result with scale factor {scale_factor} "
    f"because parallel and per-instance config match."
  )


if max_num_batched_tokens_mode not in {"fixed", "auto"}:
  raise ValueError("MAX_NUM_BATCHED_TOKENS_MODE must be fixed or auto")

ensure_container_exists()
raw_log_dir.mkdir(parents=True, exist_ok=True)

with prediction_csv.open(newline="") as handle:
  prediction_rows = [row for row in csv.DictReader(handle) if case_matches(row)]

if not prediction_rows:
  raise RuntimeError("No agg rows matched the provided filters")

prediction_rows.sort(
  key=lambda row: (
    row.get("Model", ""),
    row.get("ISL", ""),
    row.get("OSL", ""),
    row.get("TTFT_input", ""),
    row.get("TPOT_input", ""),
    row.get("Parallel", ""),
    row.get("BS", ""),
    normalize_int(row.get("# of XPUs", "") or "") or 0,
  )
)

headers = [
  "ISL",
  "OSL",
  "TTFT_input",
  "TPOT_input",
  "Model",
  "# of XPUs",
  "Parallel",
  "max_num_batched_tokens",
  "Predicted BS",
  "Predicted concurrency",
  "predicted_TTFT (ms)",
  "predicted_TPOT (ms)",
  "predicted_tokens/s/gpu",
  "predicted_tokens/s/user",
  "predicted_req/s",
  "measured_TTFT (ms)",
  "measured_TPOT (ms)",
  "measured_tokens/s/gpu",
  "measured_tokens/s/user",
  "measured_req/s",
  "delta_TTFT (ms)",
  "delta_pct_TTFT (ms)",
  "delta_TPOT (ms)",
  "delta_pct_TPOT (ms)",
  "delta_tokens/s/gpu",
  "delta_pct_tokens/s/gpu",
  "delta_tokens/s/user",
  "delta_pct_tokens/s/user",
  "delta_req/s",
  "delta_pct_req/s",
  "status",
  "notes",
]

all_rows: list[dict[str, str]] = []
derived_rows_by_key: dict[tuple[str, ...], list[dict[str, object]]] = {}
scenario_rows_by_key: dict[tuple[str, ...], list[dict[str, object]]] = {}

for row in prediction_rows:
  output_row = make_output_row(row)
  model = row["Model"]
  vllm_model_name = derive_vllm_model_name(model)
  tokenizer_name = derive_tokenizer_name(model)
  request_model_name = derive_request_model_name(model)
  num_gpus = normalize_int(row.get("# of XPUs", "") or "")
  isl = normalize_int(row.get("ISL", "") or "")
  osl = normalize_int(row.get("OSL", "") or "")
  benchmark_bs = normalize_int(row.get("concurrency", "") or "")
  parallel = row.get("Parallel", "")

  case_label = (
    f"isl{row.get('ISL','')}_osl{row.get('OSL','')}_ttft{row.get('TTFT_input','')}_"
    f"tpot{row.get('TPOT_input','')}_{sanitize_name(model)}_g{row.get('# of XPUs','')}"
  )
  benchmark_log_host = raw_log_dir / f"{case_label}_benchmark.log"
  vllm_log_host = raw_log_dir / f"{case_label}_vllm.log"

  if num_gpus is None or isl is None or osl is None or benchmark_bs is None:
    output_row["status"] = "skipped"
    output_row["notes"] = "Missing numeric fields in prediction CSV"
    log_section(f"Skipping agg case: {case_label}")
    log_kv("Reason", output_row["notes"])
    all_rows.append(output_row)
    continue

  try:
    tp = parse_tp(parallel)
  except ValueError as exc:
    output_row["status"] = "skipped"
    output_row["notes"] = str(exc)
    log_section(f"Skipping agg case: {case_label}")
    log_kv("Reason", output_row["notes"])
    all_rows.append(output_row)
    continue

  max_num_batched_tokens = isl if max_num_batched_tokens_mode == "auto" else max_num_batched_tokens_value
  output_row["max_num_batched_tokens"] = str(max_num_batched_tokens)
  derivation_key = build_derivation_key(row, max_num_batched_tokens)
  scenario_key = build_scenario_key(row, max_num_batched_tokens)
  target_concurrency = normalize_int(row.get("concurrency", "") or "")
  vllm_log_in_container = f"/tmp/aic_vllm_bench_logs/{case_label}.log"
  bench_artifact_dir = f"/tmp/aic_bench_artifacts/{case_label}/rss_agg_tp{tp}_bs{benchmark_bs}_fp16"
  bench_export_csv_in_container = f"{bench_artifact_dir}/profile_req_inf.csv"

  log_section(f"Running agg case: {case_label}")
  log_kv("Model", model)
  log_kv("GPUs", str(num_gpus))
  log_kv("Parallel", parallel)
  log_kv("Benchmark concurrency", str(benchmark_bs))
  log_kv("max_num_batched_tokens", str(max_num_batched_tokens))

  source_entry = None
  if target_concurrency is not None:
    source_entry = find_derivation_source(
      derived_rows_by_key=derived_rows_by_key,
      key=derivation_key,
      target_num_gpus=num_gpus,
      target_concurrency=target_concurrency,
    )
  if source_entry is not None:
    source_num_gpus = int(source_entry["num_gpus"])
    scale_factor = num_gpus // source_num_gpus
    log_subsection("Derivation")
    log(
      f"Derived from {source_num_gpus}-GPU case with scale factor {scale_factor}; "
      "skipping vLLM launch and AIPerf."
    )
    derive_output_row_from_source(
      output_row=output_row,
      source_row=source_entry["row"],
      scale_factor=scale_factor,
      source_num_gpus=source_num_gpus,
    )
    log_kv("Reason", output_row["notes"])
    log_measured_results(output_row)
    all_rows.append(output_row)
    derived_rows_by_key.setdefault(derivation_key, []).append(
      {"num_gpus": num_gpus, "concurrency": target_concurrency, "row": dict(output_row)}
    )
    scenario_rows_by_key.setdefault(scenario_key, []).append(
      {
        "num_gpus": num_gpus,
        "parallel": parallel,
        "bs": row.get("BS", ""),
        "concurrency": target_concurrency,
        "row": dict(output_row),
      }
    )
    continue

  if has_same_scenario_smaller_gpu_case(
    scenario_rows_by_key=scenario_rows_by_key,
    key=scenario_key,
    target_num_gpus=num_gpus,
  ):
    log(
      "Found a smaller-GPU case for the same scenario, but parallel/BS layout does not match "
      "a derivable pattern; running a real benchmark."
    )

  try:
    kill_output = kill_existing_vllm(port)
    if kill_output.strip():
      log(strip_ansi(kill_output).strip())

    pid = start_vllm(
      model=vllm_model_name,
      served_model_name=request_model_name,
      tp=tp,
      num_gpus=num_gpus,
      max_num_batched_tokens=max_num_batched_tokens,
      log_path=vllm_log_in_container,
    )
    log_subsection("vLLM Status")
    log_kv("pid", pid)

    wait_for_vllm_ready(port, startup_timeout, request_model_name)
    log_kv("ready", "yes")

    benchmark_display_command = f"""
source {shlex.quote(aiperf_venv)}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
model_name={shlex.quote(request_model_name)}
tokenizer_name={shlex.quote(tokenizer_name)}
bs={benchmark_bs}
isl={isl}
osl={osl}
aiperf profile \
  --model "$model_name" \
  --tokenizer "$tokenizer_name" \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --url http://localhost:{port} \
  --synthetic-input-tokens-mean "$isl" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$osl" \
  --output-tokens-stddev 0 \
  --extra-inputs max_tokens:$osl \
  --extra-inputs min_tokens:$osl \
  --extra-inputs ignore_eos:true \
  --streaming \
  --request-rate inf \
  --request-count $((bs * 10)) \
  --warmup-request-count $((bs * 10)) \
  --num-dataset-entries $((bs * 20)) \
  --random-seed 100 \
  --artifact-dir {shlex.quote(bench_artifact_dir)} \
  --profile-export-prefix profile_req_inf \
  --ui simple \
  --no-server-metrics \
  --concurrency "$bs"
""".strip()
    log_command("AIPerf Command", benchmark_display_command)

    benchmark_script = f"""
set -euo pipefail
source {shlex.quote(aiperf_venv)}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if ! python - <<'__AIC_CHECK_PROTOBUF__' >/dev/null 2>&1
import google.protobuf
__AIC_CHECK_PROTOBUF__
then
  extra_site_packages=$(python3 - <<'__AIC_FIND_PROTOBUF__'
import site
from pathlib import Path

for site_path in site.getsitepackages():
    candidate = Path(site_path) / 'google' / 'protobuf' / '__init__.py'
    if candidate.exists():
        print(site_path)
        break
__AIC_FIND_PROTOBUF__
)
  if [[ -n "$extra_site_packages" ]]; then
    export PYTHONPATH="$extra_site_packages${{PYTHONPATH:+:$PYTHONPATH}}"
  fi
fi
model_name={shlex.quote(request_model_name)}
tokenizer_name={shlex.quote(tokenizer_name)}
bs={benchmark_bs}
isl={isl}
osl={osl}
mkdir -p {shlex.quote(bench_artifact_dir)}
aiperf profile \
  --model "$model_name" \
  --tokenizer "$tokenizer_name" \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --url http://localhost:{port} \
  --synthetic-input-tokens-mean "$isl" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$osl" \
  --output-tokens-stddev 0 \
  --extra-inputs max_tokens:$osl \
  --extra-inputs min_tokens:$osl \
  --extra-inputs ignore_eos:true \
  --streaming \
  --request-rate inf \
  --request-count $((bs * 10)) \
  --warmup-request-count $((bs * 10)) \
  --num-dataset-entries $((bs * 20)) \
  --random-seed 100 \
  --artifact-dir {shlex.quote(bench_artifact_dir)} \
  --profile-export-prefix profile_req_inf \
  --ui simple \
  --no-server-metrics \
  --concurrency "$bs"
"""
    bench_result = run_container_script(benchmark_script, check=False)
    benchmark_output = (bench_result.stdout or "") + (bench_result.stderr or "")
    benchmark_log_host.write_text(benchmark_output)

    vllm_log_host.write_text(fetch_container_file(vllm_log_in_container))

    if bench_result.returncode != 0:
      output_row["status"] = "benchmark_failed"
      output_row["notes"] = f"aiperf exited with code {bench_result.returncode}"
      log_benchmark_failure(bench_result.returncode, benchmark_output)
      all_rows.append(output_row)
      continue

    export_csv_text = fetch_container_file(bench_export_csv_in_container)
    metrics = parse_metrics_from_export(export_csv_text, num_gpus=num_gpus)
    if any(not value for value in metrics.values()):
      fallback_metrics = parse_metrics(benchmark_output, num_gpus=num_gpus, osl=osl)
      for key, value in fallback_metrics.items():
        if not metrics.get(key) and value:
          metrics[key] = value
    output_row.update(metrics)
    add_comparison_metrics(output_row)
    missing_metrics = [key for key, value in metrics.items() if not value]
    output_row["status"] = "success" if not missing_metrics else "partial_success"
    if missing_metrics:
      output_row["notes"] = "Missing metrics: " + ", ".join(missing_metrics)
      log_subsection("Warnings")
      log(output_row["notes"])
    log_measured_results(output_row)
    all_rows.append(output_row)
    if target_concurrency is not None and has_complete_measured_metrics(output_row):
      derived_rows_by_key.setdefault(derivation_key, []).append(
        {"num_gpus": num_gpus, "concurrency": target_concurrency, "row": dict(output_row)}
      )
    scenario_rows_by_key.setdefault(scenario_key, []).append(
      {
        "num_gpus": num_gpus,
        "parallel": parallel,
        "bs": row.get("BS", ""),
        "concurrency": target_concurrency,
        "row": dict(output_row),
      }
    )
  except Exception as exc:
    output_row["status"] = "failed"
    output_row["notes"] = str(exc)
    log_subsection("Case Failure")
    log(str(exc))
    try:
      vllm_log_host.write_text(fetch_container_file(vllm_log_in_container))
    except Exception:
      pass
    all_rows.append(output_row)
  finally:
    try:
      kill_existing_vllm(port)
    except Exception:
      pass

with output_csv.open("w", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=headers)
  writer.writeheader()
  writer.writerows(all_rows)

log_section("Run Complete")
log_kv("rows saved", str(len(all_rows)))
log_kv("output csv", str(output_csv))
log_kv("raw logs", str(raw_log_dir))
PY