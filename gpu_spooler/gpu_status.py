#!/usr/bin/env python3
"""
GPU Server Status Checker for VUW ECS GPU Servers

SSHes to each server in parallel, runs nvidia-smi, and displays a table
showing free/busy GPUs, users, and running processes.

Usage:
    python3 gpu_status.py [--detail] [--free-only] [--server <name>]

Options:
    --detail      Show per-GPU rows (default: show server summary)
    --free-only   Only show servers/GPUs with free capacity
    --server      Check a single server by name
"""

import subprocess
import concurrent.futures
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Server catalogue
# ---------------------------------------------------------------------------

SERVERS = [
    "cuda-small0", "cuda-small1",
    "cuda00", "cuda01", "cuda02", "cuda03", "cuda04", "cuda05", "cuda06",
    "cuda07", "cuda08", "cuda09", "cuda10", "cuda11", "cuda12", "cuda13",
    "cuda14", "cuda15", "cuda16", "cuda17", "cuda18", "cuda19", "cuda20",
    "cuda21", "cuda22",
    "gryphon", "red-tomatoes", "piccolo", "the-villa", "bordeaux",
]

SERVER_INFO = {
    "cuda-small0":  {"gpu_type": "RTX A4000",    "gpu_mem_gb": 16, "cpu_cores": 16,  "sys_mem_gb": 192},
    "cuda-small1":  {"gpu_type": "RTX 4000 Ada", "gpu_mem_gb": 20, "cpu_cores": 16,  "sys_mem_gb": 144},
    "cuda00":       {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 32,  "sys_mem_gb": 96},
    "cuda01":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 16,  "sys_mem_gb": 64},
    "cuda02":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 16,  "sys_mem_gb": 64},
    "cuda03":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 16,  "sys_mem_gb": 64},
    "cuda04":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 16,  "sys_mem_gb": 64},
    "cuda05":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 144},
    "cuda06":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 16,  "sys_mem_gb": 144},
    "cuda07":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 144},
    "cuda08":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 144},
    "cuda09":       {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 32,  "sys_mem_gb": 96},
    "cuda10":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 128},
    "cuda11":       {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 112},
    "cuda12":       {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 32,  "sys_mem_gb": 96},
    "cuda13":       {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 32,  "sys_mem_gb": 96},
    "cuda14":       {"gpu_type": "A40",          "gpu_mem_gb": 48, "cpu_cores": 64,  "sys_mem_gb": 256},
    "cuda15":       {"gpu_type": "A40",          "gpu_mem_gb": 48, "cpu_cores": 64,  "sys_mem_gb": 256},
    "cuda16":       {"gpu_type": "RTX A6000",    "gpu_mem_gb": 48, "cpu_cores": 32,  "sys_mem_gb": 192},
    "cuda17":       {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 24,  "sys_mem_gb": 144},
    "cuda18":       {"gpu_type": "RTX A6000",    "gpu_mem_gb": 48, "cpu_cores": 32,  "sys_mem_gb": 192},
    "cuda19":       {"gpu_type": "L4",           "gpu_mem_gb": 24, "cpu_cores": 64,  "sys_mem_gb": 256},
    "cuda20":       {"gpu_type": "L40S",         "gpu_mem_gb": 48, "cpu_cores": 64,  "sys_mem_gb": 384},
    "cuda21":       {"gpu_type": "RTX 6000 Ada", "gpu_mem_gb": 48, "cpu_cores": 48,  "sys_mem_gb": 256},
    "cuda22":       {"gpu_type": "RTX 6000 Ada", "gpu_mem_gb": 48, "cpu_cores": 48,  "sys_mem_gb": 256},
    "gryphon":      {"gpu_type": "RTX A6000",    "gpu_mem_gb": 48, "cpu_cores": 64,  "sys_mem_gb": 192},
    "red-tomatoes": {"gpu_type": "RTX A6000",    "gpu_mem_gb": 48, "cpu_cores": 32,  "sys_mem_gb": 96},
    "piccolo":      {"gpu_type": "RTX A5000",    "gpu_mem_gb": 24, "cpu_cores": 20,  "sys_mem_gb": 128},
    "the-villa":    {"gpu_type": "RTX A6000",    "gpu_mem_gb": 48, "cpu_cores": 64,  "sys_mem_gb": 192},
    "bordeaux":     {"gpu_type": "RTX 6000",     "gpu_mem_gb": 24, "cpu_cores": 20,  "sys_mem_gb": 64},
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GpuProcess:
    pid: int
    user: str
    mem_used_mb: int
    command: str  # truncated command/script name

@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_free_mb: int
    mem_total_mb: int
    utilization_pct: int          # -1 if unknown
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        # A GPU is free if no processes are listed AND VRAM usage is very low.
        # (Some zombie contexts might hold VRAM but not show in pmon).
        return len(self.processes) == 0 and self.mem_used_mb < 128

@dataclass
class ServerResult:
    name: str
    reachable: bool
    error: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def free_gpus(self) -> int:
        return sum(1 for g in self.gpus if g.is_free)

    @property
    def busy_gpus(self) -> int:
        return self.total_gpus - self.free_gpus

    @property
    def total_vram_mb(self) -> int:
        return sum(g.mem_total_mb for g in self.gpus)

    @property
    def free_vram_mb(self) -> int:
        return sum(g.mem_free_mb for g in self.gpus)

# ---------------------------------------------------------------------------
# Remote data collection
# ---------------------------------------------------------------------------

# Single bash command run remotely; sections delimited by sentinel lines.
REMOTE_CMD = r"""
GPU_OUT=$(nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits 2>/dev/null) || { echo "ERROR: nvidia-smi failed"; exit 1; }

PMON_OUT=$(nvidia-smi pmon -c 1 2>/dev/null)
APPS_OUT=$(nvidia-smi --query-compute-apps=gpu_index,pid --format=csv,noheader 2>/dev/null)

printf 'GPU_INFO\n%s\nPMON_INFO\n%s\nAPPS_INFO\n%s\nPROC_INFO\n' "$GPU_OUT" "$PMON_OUT" "$APPS_OUT"
# Extract unique PIDs from pmon and apps
PIDS=$(printf "%s\n%s" "$PMON_OUT" "$APPS_OUT" | awk 'NR>2 && $2~/^[0-9]+$/{print $2}' | sort -u | tr '\n' ',')
PIDS="${PIDS%,}"
if [ -n "$PIDS" ]; then
    ps -o pid=,user= -p "$PIDS" 2>/dev/null || true
fi
printf 'END\n'
"""


def ssh_run(server: str, timeout: int = 20, debug: bool = False) -> tuple[bool, str]:
    """SSH to server and run the remote collection script."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",          # never prompt for password
                "-o", "LogLevel=ERROR",
                server,
                REMOTE_CMD,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if debug:
            print(f"\n--- RAW OUTPUT FROM {server} ---")
            print(result.stdout or "(empty stdout)")
            if result.stderr:
                print(f"STDERR: {result.stderr.strip()}")
            print(f"--- END {server} (exit={result.returncode}) ---\n")
        if result.returncode != 0 and not result.stdout:
            return False, result.stderr.strip() or f"ssh exited {result.returncode}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "SSH timed out"
    except FileNotFoundError:
        return False, "ssh binary not found"


def parse_output(raw: str) -> list[GpuInfo]:
    """Parse the structured output from REMOTE_CMD into GpuInfo objects."""
    SENTINELS = {"GPU_INFO", "PMON_INFO", "APPS_INFO", "PROC_INFO"}
    sections: dict[str, list[str]] = {s: [] for s in SENTINELS}
    current = None
    for line in raw.splitlines():
        stripped = line.rstrip("\r")          # tolerate \r\n from some SSH configs
        if stripped in SENTINELS:
            current = stripped
        elif stripped == "END":
            break
        elif current and stripped.strip():
            sections[current].append(stripped)

    # --- GPU rows (CSV from --query-gpu) ---
    gpus: dict[int, GpuInfo] = {}
    for line in sections["GPU_INFO"]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            util_str = parts[5]
            util = int(util_str) if util_str.lstrip("-").isdigit() else -1
            gpus[idx] = GpuInfo(
                index=idx,
                name=parts[1],
                mem_used_mb=int(parts[2]),
                mem_free_mb=int(parts[3]),
                mem_total_mb=int(parts[4]),
                utilization_pct=util,
            )
        except (ValueError, IndexError):
            continue

    # --- pmon and apps info ---
    # pid -> (gpu_idx, fb_mb, cmd_name)
    pid_to_gpu: dict[int, tuple[int, int, str]] = {}
    
    for line in sections["PMON_INFO"]:
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        if not parts[1].isdigit():
            continue
        try:
            gpu_idx = int(parts[0])
            pid     = int(parts[1])
            fb_mb   = int(parts[7]) if parts[7].isdigit() else 0
            cmd     = parts[8]
            pid_to_gpu[pid] = (gpu_idx, fb_mb, cmd)
        except (ValueError, IndexError):
            continue

    for line in sections["APPS_INFO"]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu_idx = int(parts[0])
            pid     = int(parts[1])
            if pid not in pid_to_gpu:
                pid_to_gpu[pid] = (gpu_idx, 0, "unknown")
        except (ValueError, IndexError):
            continue

    # --- proc info rows: pid user ---
    pid_user: dict[int, str] = {}
    for line in sections["PROC_INFO"]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                pid_user[int(parts[0])] = parts[1]
            except ValueError:
                continue

    # --- attach processes to GPUs ---
    for pid, (gpu_idx, fb_mb, cmd) in pid_to_gpu.items():
        if gpu_idx not in gpus:
            continue
        user = pid_user.get(pid, "?")
        gpus[gpu_idx].processes.append(GpuProcess(
            pid=pid,
            user=user,
            mem_used_mb=fb_mb,
            command=cmd[:60],
        ))

    return list(gpus.values())


def query_server(server: str, debug: bool = False) -> ServerResult:
    ok, output = ssh_run(server, debug=debug)
    if not ok:
        return ServerResult(name=server, reachable=False, error=output)
    if output.startswith("ERROR:"):
        return ServerResult(name=server, reachable=False, error=output.strip())
    gpus = parse_output(output)
    return ServerResult(name=server, reachable=True, gpus=gpus)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
ORANGE = "\033[33m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def mb_to_gb(mb: int) -> str:
    return f"{mb / 1024:.1f}"


def bar(used_mb: int, total_mb: int, width: int = 10) -> str:
    """Simple ASCII utilisation bar."""
    if total_mb == 0:
        return " " * width
    frac = used_mb / total_mb
    filled = round(frac * width)
    bar_str = "█" * filled + "░" * (width - filled)
    color = GREEN if frac < 0.5 else (YELLOW if frac < 0.85 else RED)
    return f"{color}{bar_str}{RESET}"


def print_summary_table(results: list[ServerResult], free_only: bool = False) -> None:
    """One row per server."""
    header = (
        f"{'SERVER':<14} {'GPU TYPE':<14} {'GPUs':>4}  {'FREE':>4}  "
        f"{'VRAM FREE':>10}  {'VRAM TOTAL':>10}  {'USERS / PROCESSES'}"
    )
    sep = "─" * len(header)
    print(f"\n{BOLD}{header}{RESET}")
    print(sep)

    for r in sorted(results, key=lambda x: (not x.reachable, x.name)):
        if free_only and r.reachable and r.free_gpus == 0:
            continue

        info = SERVER_INFO.get(r.name, {})
        gpu_type = info.get("gpu_type", "?")

        if not r.reachable:
            print(
                f"{r.name:<14} {colorize('UNREACHABLE', RED):<23}  {r.error[:60]}"
            )
            continue

        total = r.total_gpus
        free  = r.free_gpus

        if free == total:
            free_str = colorize(str(free), GREEN)
        elif free == 0:
            free_str = colorize(str(free), RED)
        else:
            free_str = colorize(str(free), YELLOW)

        # Collect unique users and their processes
        user_procs: dict[str, set[str]] = {}
        for g in r.gpus:
            for p in g.processes:
                user_procs.setdefault(p.user, set()).add(p.command)

        who = "; ".join(
            f"{colorize(u, CYAN)}: {', '.join(sorted(cmds))}"
            for u, cmds in sorted(user_procs.items())
        ) or colorize("idle", DIM)

        vram_free  = f"{mb_to_gb(r.free_vram_mb):>5} GB"
        vram_total = f"{mb_to_gb(r.total_vram_mb):>5} GB"

        print(
            f"{r.name:<14} {gpu_type:<14} {total:>4}  {free_str:>12}  "
            f"{vram_free:>10}  {vram_total:>10}  {who}"
        )

    print(sep)


def print_detail_table(results: list[ServerResult], free_only: bool = False) -> None:
    """One row per GPU."""
    header = (
        f"{'SERVER':<14} {'GPU':>3}  {'GPU NAME':<20}  "
        f"{'MEM USED':>9}  {'MEM TOT':>7}  {'UTIL':>5}  "
        f"{'STATUS':<8}  {'USER':<12}  {'PROCESS / SCRIPT'}"
    )
    sep = "─" * len(header)
    print(f"\n{BOLD}{header}{RESET}")
    print(sep)

    for r in sorted(results, key=lambda x: (not x.reachable, x.name)):
        if not r.reachable:
            print(f"{r.name:<14} {colorize('UNREACHABLE', RED)}  {r.error[:60]}")
            continue

        for g in sorted(r.gpus, key=lambda x: x.index):
            if free_only and not g.is_free:
                continue

            mem_used_str = f"{mb_to_gb(g.mem_used_mb):>6} GB"
            mem_tot_str  = f"{mb_to_gb(g.mem_total_mb):>4} GB"
            util_str = f"{g.utilization_pct:>3}%" if g.utilization_pct >= 0 else "  N/A"

            if g.is_free:
                status = colorize("FREE", GREEN)
                user_col = ""
                proc_col = ""
                print(
                    f"{r.name:<14} {g.index:>3}  {g.name:<20}  "
                    f"{mem_used_str:>9}  {mem_tot_str:>7}  {util_str:>5}  "
                    f"{status:<16}  {user_col:<12}  {proc_col}"
                )
            else:
                status = colorize("IN USE", RED)
                # Print one line per process (first process on same row as GPU)
                procs = g.processes if g.processes else [GpuProcess(0, "?", 0, "?")]
                first = True
                for p in procs:
                    if first:
                        print(
                            f"{r.name:<14} {g.index:>3}  {g.name:<20}  "
                            f"{mem_used_str:>9}  {mem_tot_str:>7}  {util_str:>5}  "
                            f"{status:<16}  {colorize(p.user, CYAN):<20}  {p.command}"
                        )
                        first = False
                    else:
                        print(
                            f"{'':14} {'':>3}  {'':20}  "
                            f"{'':>9}  {'':>7}  {'':>5}  "
                            f"{'':16}  {colorize(p.user, CYAN):<20}  {p.command}"
                        )

    print(sep)


def print_totals(results: list[ServerResult]) -> None:
    reachable = [r for r in results if r.reachable]
    total_gpus  = sum(r.total_gpus for r in reachable)
    free_gpus   = sum(r.free_gpus  for r in reachable)
    total_vram  = sum(r.total_vram_mb for r in reachable)
    free_vram   = sum(r.free_vram_mb  for r in reachable)
    unreachable = len(results) - len(reachable)

    print(
        f"\n{BOLD}Totals:{RESET}  "
        f"{colorize(str(free_gpus), GREEN)} / {total_gpus} GPUs free  |  "
        f"{mb_to_gb(free_vram)} / {mb_to_gb(total_vram)} GB VRAM free  |  "
        f"{colorize(str(unreachable), RED) if unreachable else colorize('0', GREEN)} servers unreachable\n"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GPU availability across VUW ECS GPU servers."
    )
    parser.add_argument(
        "--detail", "-d", action="store_true",
        help="Show per-GPU detail rows instead of per-server summary"
    )
    parser.add_argument(
        "--free-only", "-f", action="store_true",
        help="Only show servers / GPUs with free capacity"
    )
    parser.add_argument(
        "--server", "-s", metavar="NAME",
        help="Query a single server by hostname"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=16,
        help="Number of parallel SSH workers (default: 16)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw SSH output for each server (useful for diagnosing parse issues)"
    )
    args = parser.parse_args()

    targets = [args.server] if args.server else SERVERS

    print(f"{BOLD}Querying {len(targets)} server(s)...{RESET}", flush=True)

    results: list[ServerResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(query_server, s, args.debug): s for s in targets}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            server = futures[future]
            result = future.result()
            results.append(result)
            # Live progress indicator
            status = "ok" if result.reachable else f"ERR: {result.error[:40]}"
            print(f"  [{i:>2}/{len(targets)}] {server:<16} {status}", flush=True)

    if args.detail:
        print_detail_table(results, free_only=args.free_only)
    else:
        print_summary_table(results, free_only=args.free_only)

    print_totals(results)


if __name__ == "__main__":
    main()
