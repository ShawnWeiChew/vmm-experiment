"""
Build and run the vmm-experiments binary on Modal, on a multi-GPU (NVLink/NVSwitch)
box so `cuMulticastCreate` has GPUs to bind.

Usage:
    modal run modal_run.py
"""

import modal

app = modal.App("vmm-experiments")

CUDA_TAG = "12.4.1-devel-ubuntu22.04"  # multicast VMM API needs CUDA >= 12.4

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("build-essential")
    .add_local_dir(".", remote_path="/root/vmm-experiments")
)

NUM_GPUS = 2  # matches num_threads in vmm/main.cpp


@app.function(image=image, gpu=f"H100:{NUM_GPUS}", timeout=600)
def build_and_run():
    import subprocess

    repo = "/root/vmm-experiments"

    subprocess.run(["nvidia-smi", "-L"], check=True)

    subprocess.run(
        ["make", "-B"],
        cwd=repo,
        check=True,
        env={"PATH": "/usr/local/cuda/bin:/usr/bin:/bin"},
    )

    result = subprocess.run(
        ["./main"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    print("--- stdout ---")
    print(result.stdout)
    print("--- stderr ---")
    print(result.stderr)
    return result.returncode


@app.local_entrypoint()
def main():
    rc = build_and_run.remote()
    print(f"exit code: {rc}")
    if rc != 0:
        raise SystemExit(rc)
