# gpu-spooler

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey?logo=linux&logoColor=white)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Servers](https://img.shields.io/badge/GPU%20servers-30-blueviolet?logo=nvidia&logoColor=white)](https://www.ecs.vuw.ac.nz)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com)

A distributed GPU task spooler for the **VUW ECS GPU cluster**. Queue jobs from anywhere, and the background daemon automatically finds a server with free GPUs, SSHes in, and runs them — no manual server hunting required.

---

## Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Submitting Tasks](#submitting-tasks)
  - [Inspecting Tasks](#inspecting-tasks)
  - [Managing Tasks](#managing-tasks)
  - [GPU Status](#gpu-status)
  - [Cluster Operations](#cluster-operations)
  - [Daemon](#daemon)
- [GPU Servers](#gpu-servers)
- [State & Logs](#state--logs)
- [Troubleshooting](#troubleshooting)

---

## How It Works

```
 You                  Daemon                    GPU Servers
  │                     │                           │
  ├─ task -G 1 cmd ──►  │                           │
  │  (queued in DB)      │                           │
  │                      ├── SSH → nvidia-smi ──►   │
  │                      │◄── free GPUs ─────────── │
  │                      │                           │
  │                      ├── SSH → bash -l -c ───►  │
  │                      │   CUDA_VISIBLE_DEVICES=0  │
  │                      │   cmd                     │
  │                      │                           │
  ├─ task -o 1 ───────►  │◄── stdout/stderr ──────── │
  │◄── output ──────────  │                           │
```

1. `task -G N cmd` adds a job to a local SQLite queue and auto-starts the daemon.
2. The daemon polls all GPU servers in parallel every 30 seconds using `nvidia-smi pmon`.
3. When a server has ≥ N free GPUs, the daemon SSHes in, sets `CUDA_VISIBLE_DEVICES`, and runs the command in a login shell (so your `~/.bashrc` and conda environments are loaded).
4. Output is streamed back and saved to `~/.task-spooler/logs/<id>.log`.
5. The daemon uses **bin-packing** — it fills busier servers first to avoid fragmenting GPU resources across too many machines.

---

## Installation

**Requirements:** Python 3.10+, SSH access to the ECS GPU servers with key-based auth (no password prompts).

```bash
git clone <repo-url> task-spooler
cd task-spooler
bash install.sh
```

The installer creates a `.venv` in the project directory, installs the package in editable mode, and symlinks `task` and `gpu-status` into `~/.local/bin`.

If `~/.local/bin` is not in your `PATH`, add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Since the install is **editable**, any changes you make to the source files take effect immediately without reinstalling.

### Installing packages across the cluster

To install a Python package on every GPU server at once (e.g. after adding a new dependency):

```bash
task -R "cd /vol/ecrg-solar/woodj4/depth-learning && pip install -e ."
```

---

## Quick Start

```bash
# Check which servers have free GPUs
gpu-status

# Queue a training job requiring 1 GPU
task -G 1 python3 train.py --epochs 50

# Queue a job requiring 2 GPUs with a label
task -G 2 -n "ablation run" python3 train.py --model large

# Watch the queue
task -l

# Follow live output
task -f 1

# Kill everything and reset
task -K
```

---

## Commands

### Submitting Tasks

```
task -G <N> [options] <command...>
```

| Flag | Description |
|------|-------------|
| `-G N` / `--gpus N` | **Required.** Number of GPUs the job needs. |
| `-n LABEL` / `--name LABEL` | Optional human-readable label shown in the task list. |
| `-d PATH` / `--dir PATH` | Working directory on the remote server (default: current directory). Useful when the local and remote mount paths differ. |

**Examples:**

```bash
task -G 1 python3 train.py --task counting --dataset easy --epochs 100
task -G 2 -n "multi-gpu run" bash experiments/run.sh
task -G 1 -d /vol/ecrg-solar/woodj4/project python3 eval.py
```

> Arguments after the command — including flags like `--task` and `--dataset` — are passed through verbatim and never interpreted by the spooler.

---

### Inspecting Tasks

| Command | Description |
|---------|-------------|
| `task -l` / `task --list` | List all tasks with status, server, GPU indices, runtime, and label. |
| `task -i ID` / `task --info ID` | Detailed info for a single task: server, GPU type, VRAM, working dir, timestamps, exit code, log path. |
| `task -o ID` / `task --output ID` | Print the full saved output (stdout + stderr) of a task. |
| `task -f ID` / `task --follow ID` | Stream live output as the task runs (`tail -f` style). Waits if the task hasn't started yet. |

**Task status colours:**

| Status | Meaning |
|--------|---------|
| `queued` | Waiting for a free GPU |
| `running` | Currently executing on a server |
| `done` | Finished successfully (exit 0) |
| `failed` | Finished with a non-zero exit code |
| `killed` | Manually killed via `task -k` or `task -K` |
| `cancelled` | Cancelled before it started |
| `lost` | Was running when the daemon was restarted |

---

### Managing Tasks

| Command | Description |
|---------|-------------|
| `task -k ID` / `task --kill ID` | Kill a running task (SIGTERM to SSH client) or cancel a queued one. |
| `task -K` / `task --kill-all` | Kill all running tasks, cancel all queued tasks, stop the daemon, clear the task list, and reset the ID counter to 1. |
| `task -c` / `task --clear` | Remove finished tasks (done/failed/killed/cancelled) from the list without stopping anything. |
| `task -w ID` / `task --wait ID` | Block until the specified task finishes. Useful in scripts. |

---

### GPU Status

```bash
gpu-status                  # per-server summary table
gpu-status --detail         # one row per GPU
gpu-status --free-only      # hide fully-busy servers
gpu-status --server cuda14  # check a single server
gpu-status --debug          # print raw SSH output (for diagnosing issues)
```

You can also check GPU status from within the task command:

```bash
task --gpu-status
```

**Example output:**

```
SERVER          GPU TYPE        GPUs  FREE   VRAM FREE   VRAM TOTAL  USERS / PROCESSES
──────────────────────────────────────────────────────────────────────────────────────
cuda14          A40                4     2     96.0 GB    192.0 GB   woodj4: python3 train.py
cuda20          L40S               4     4    192.0 GB    192.0 GB   idle
cuda16          RTX A6000          3     0      0.0 GB    144.0 GB   jsmith: bash run.sh
...
```

---

### Cluster Operations

Run any shell command on **every GPU server in parallel**:

```bash
task -R "CMD"
```

Results stream in as servers respond, with a live progress indicator. Failed servers print their full output at the end.

**Examples:**

```bash
# Install a package everywhere
task -R "cd /vol/ecrg-solar/woodj4/depth-learning && pip install -e ."

# Check Python version on all servers
task -R "python3 --version"

# Check disk usage on the shared volume
task -R "df -h /vol/ecrg-solar"
```

---

### Daemon

The daemon starts automatically when you submit a job. You rarely need to manage it directly.

| Command | Description |
|---------|-------------|
| `task --daemon` | Start the daemon if it isn't already running. |
| `task --stop` | Gracefully stop the daemon (running tasks continue on the server but are no longer monitored). |

Daemon logs are written to `~/.task-spooler/daemon.log`.

---

## GPU Servers

| Server | GPUs | Type | VRAM/GPU | CPU Cores | RAM |
|--------|-----:|------|----------:|----------:|----:|
| cuda-small0 | 4 | RTX A4000 | 16 GB | 16 | 192 GB |
| cuda-small1 | 3 | RTX 4000 Ada | 20 GB | 16 | 144 GB |
| cuda00 | 2 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda01–04 | 2 | RTX A5000 | 24 GB | 16 | 64 GB |
| cuda05–08 | 3 | RTX A5000 | 24 GB | 24 | 144 GB |
| cuda09 | 3 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda10–11 | 3 | RTX A5000 | 24 GB | 24 | 112–128 GB |
| cuda12–13 | 3 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda14–15 | 4 | A40 | 48 GB | 64 | 256 GB |
| cuda16 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda17 | 3 | RTX 6000 | 24 GB | 24 | 144 GB |
| cuda18 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda19 | 4 | L4 | 24 GB | 64 | 256 GB |
| cuda20 | 4 | L40S | 48 GB | 64 | 384 GB |
| cuda21–22 | 2 | RTX 6000 Ada | 48 GB | 48 | 256 GB |
| gryphon | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| red-tomatoes | 2 | RTX A6000 | 48 GB | 32 | 96 GB |
| piccolo | 2 | RTX A5000 | 24 GB | 20 | 128 GB |
| the-villa | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| bordeaux | 1 | RTX 6000 | 24 GB | 20 | 64 GB |

**Total: 30 servers · 80 GPUs**

---

## State & Logs

All state is stored in `~/.task-spooler/`:

```
~/.task-spooler/
├── tasks.db        # SQLite database (all task records)
├── daemon.pid      # PID of the running daemon
├── daemon.log      # Daemon stdout/stderr
└── logs/
    ├── 1.log       # Output for task 1
    ├── 2.log       # Output for task 2
    └── ...
```

Task logs include a header with the command, server, GPU indices, start time, and working directory, followed by the raw stdout/stderr of your job.

---

## Troubleshooting

**Command not found on the remote server**

The daemon sources `~/.bashrc` before running your command, which should load conda environments and `~/.local/bin`. If a command is still missing, verify it's accessible in a non-interactive SSH session:

```bash
ssh cuda14 "bash -l -c 'source ~/.bashrc; which mycommand'"
```

**Working directory not found (`No such file or directory`)**

NFS automount paths (e.g. `/am/...`) may not be mounted yet when the SSH session starts. The spooler runs `ls <dir>` first to trigger the automount. If this still fails, use `-d` to provide the path as seen from the remote server:

```bash
task -G 1 -d /vol/ecrg-solar/woodj4/project python3 train.py
```

**All servers show as unreachable**

SSH key-based auth must be set up. Test with:

```bash
ssh -o BatchMode=yes cuda01 echo ok
```

If this prompts for a password, add your public key to the servers:

```bash
ssh-copy-id cuda01
```

**GPU shows as busy but no jobs are running**

`gpu-status` uses `nvidia-smi pmon` which shows all processes using the GPU, including display servers and idle CUDA contexts. Run `gpu-status --detail` to see the specific process names and users holding each GPU.

**Daemon disappeared while a task was running**

Tasks whose daemon died are marked as `lost` on the next daemon start. The job may still be running on the remote server (since it was started with SSH keepalive). Check with:

```bash
ssh <server> "ps aux | grep mycommand"
```
