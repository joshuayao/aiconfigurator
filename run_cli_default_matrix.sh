#!/usr/bin/env bash
set -euo pipefail

# Run the default-mode matrix and collect Rank 1 rows from both:
# - agg Top Configurations
# - disagg Top Configurations
#
# Output:
# - CSV summary: cli_default_matrix_results.csv
# - Raw per-case logs: cli_default_matrix_logs/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_TSV:-$ROOT_DIR/cli_default_matrix_results.csv}}"
RAW_LOG_DIR="${RAW_LOG_DIR:-$ROOT_DIR/cli_default_matrix_logs}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/myenv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/myenv/bin/python"
else
  PYTHON_CMD="python"
fi

mkdir -p "$RAW_LOG_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_CMD" - "$ROOT_DIR" "$OUTPUT_CSV" "$RAW_LOG_DIR" <<'PY'
import csv
import re
import shlex
from pathlib import Path
import subprocess
import sys

root_dir = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
raw_log_dir = Path(sys.argv[3])

models = [
  "Qwen/Qwen3-8B",
  "/home/yaoyi/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
]
total_gpus_list = [2, 4]
scenarios = [
  (1500, 150, 500, 50),
  (2000, 256, 1500, 50),
  (4500, 1000, 2000, 50),
  (8192, 1024, 6000, 30),
]

headers = [
  "ISL",
  "OSL",
  "TTFT_input",
  "TPOT_input",
  "Model",
  "# of XPUs",
  "P/D",
  "XpYd",
  "Parallel",
  "BS",
  "ctx_tokens",
  "correction_factor",
  "concurrency",
  "TTFT (ms)",
  "TPOT (ms)",
  "tokens/s/gpu",
  "tokens/s/user",
  "req/s",
]

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
  return ANSI_ESCAPE_RE.sub("", text)


def split_table_row(line):
  if not line.startswith("|"):
    return []
  return [strip_ansi(cell).strip() for cell in line.split("|")[1:-1]]


def parse_rank1_row(output_text, section_title):
  lines = output_text.splitlines()
  section_index = None
  for index, line in enumerate(lines):
    if section_title in line:
      section_index = index
      break
  if section_index is None:
    return None

  header = None
  for line in lines[section_index + 1 :]:
    if line.startswith("+---"):
      continue
    if line.startswith("|") and "Rank" in line:
      header = split_table_row(line)
      continue
    if line.startswith("|") and header:
      cells = split_table_row(line)
      if cells and cells[0] == "1":
        return dict(zip(header, cells))
  return None


def make_empty_row(model, total_gpus, isl, osl, ttft, tpot, is_pd):
  return {
    "ISL": str(isl),
    "OSL": str(osl),
    "TTFT_input": str(ttft),
    "TPOT_input": str(tpot),
    "Model": model,
    "# of XPUs": str(total_gpus),
    "P/D": "disagg" if is_pd else "agg",
    "XpYd": "" if is_pd else "n/a",
    "Parallel": "",
    "BS": "",
    "ctx_tokens": "",
    "correction_factor": "",
    "concurrency": "",
    "TTFT (ms)": "",
    "TPOT (ms)": "",
    "tokens/s/gpu": "",
    "tokens/s/user": "",
    "req/s": "",
  }


def make_agg_row(model, total_gpus, isl, osl, ttft, tpot, row):
  return {
    "ISL": str(isl),
    "OSL": str(osl),
    "TTFT_input": str(ttft),
    "TPOT_input": str(tpot),
    "Model": model,
    "# of XPUs": str(total_gpus),
    "P/D": "agg",
    "XpYd": "n/a",
    "Parallel": row.get("parallel", ""),
    "BS": row.get("bs", ""),
    "ctx_tokens": row.get("ctx_tokens", ""),
    "correction_factor": row.get("correction_factor", ""),
    "concurrency": row.get("concurrency", ""),
    "TTFT (ms)": row.get("TTFT", ""),
    "TPOT (ms)": row.get("TPOT", ""),
    "tokens/s/gpu": row.get("tokens/s/gpu", ""),
    "tokens/s/user": row.get("tokens/s/user", ""),
    "req/s": row.get("req/s", ""),
  }


def make_disagg_row(model, total_gpus, isl, osl, ttft, tpot, row):
  p_workers = row.get("(p)workers", "")
  d_workers = row.get("(d)workers", "")
  xpyd = f"{p_workers}P{d_workers}D" if p_workers and d_workers else ""
  parallel = ""
  if row.get("(p)parallel") or row.get("(d)parallel"):
    parallel = f"(p) {row.get('(p)parallel', '')}; (d) {row.get('(d)parallel', '')}"
  bs = ""
  if row.get("(p)bs") or row.get("(d)bs"):
    bs = f"(p) {row.get('(p)bs', '')}; (d) {row.get('(d)bs', '')}"
  return {
    "ISL": str(isl),
    "OSL": str(osl),
    "TTFT_input": str(ttft),
    "TPOT_input": str(tpot),
    "Model": model,
    "# of XPUs": str(total_gpus),
    "P/D": "disagg",
    "XpYd": xpyd,
    "Parallel": parallel,
    "BS": bs,
    "ctx_tokens": "",
    "correction_factor": "",
    "concurrency": row.get("concurrency", ""),
    "TTFT (ms)": row.get("TTFT", ""),
    "TPOT (ms)": row.get("TPOT", ""),
    "tokens/s/gpu": row.get("tokens/s/gpu", ""),
    "tokens/s/user": row.get("tokens/s/user", ""),
    "req/s": row.get("req/s", ""),
  }


all_rows = []
command_cwd = root_dir / "src" / "aiconfigurator"

for isl, osl, ttft, tpot in scenarios:
  for model in models:
    for total_gpus in total_gpus_list:
      cmd = [
        sys.executable,
        "main.py",
        "cli",
        "default",
        "--model",
        model,
        "--total-gpus",
        str(total_gpus),
        "--system",
        "b60",
        "--backend",
        "vllm",
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--enable-chunked-prefill",
        "--ttft",
        str(ttft),
        "--tpot",
        str(tpot),
      ]
      display_cmd = ["python", *cmd[1:]]
      print(f">>> Running: {shlex.join(display_cmd)}", flush=True)

      result = subprocess.run(
        cmd,
        cwd=command_cwd,
        capture_output=True,
        text=True,
        env={**dict(), **__import__("os").environ, "PYTHONPATH": str(root_dir / "src")},
      )
      output_text = (result.stdout or "") + (result.stderr or "")
      safe_model = model.replace("/", "_").replace(" ", "_")
      log_path = raw_log_dir / f"isl{isl}_osl{osl}_ttft{ttft}_tpot{tpot}_{safe_model}_g{total_gpus}.log"
      log_path.write_text(output_text)

      if output_text:
        print(output_text, end="" if output_text.endswith("\n") else "\n")

      agg_row = parse_rank1_row(output_text, "agg Top Configurations:")
      if agg_row is not None:
        all_rows.append(make_agg_row(model, total_gpus, isl, osl, ttft, tpot, agg_row))
      else:
        all_rows.append(make_empty_row(model, total_gpus, isl, osl, ttft, tpot, is_pd=False))

      disagg_row = parse_rank1_row(output_text, "disagg Top Configurations:")
      if disagg_row is not None:
        all_rows.append(make_disagg_row(model, total_gpus, isl, osl, ttft, tpot, disagg_row))
      else:
        all_rows.append(make_empty_row(model, total_gpus, isl, osl, ttft, tpot, is_pd=True))


with output_csv.open("w", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=headers)
  writer.writeheader()
  writer.writerows(all_rows)

print(f"Saved {len(all_rows)} rows to {output_csv}")
print(f"Saved raw logs under {raw_log_dir}")
PY

echo "All matrix runs completed."
