# CycleGAN Style Transfer — Phase 8

Translate synthetic breadboard renders into photorealistic images while
preserving component positions so the existing bounding-box annotations
remain valid.

This directory holds CycleGAN-specific code and instructions. The actual
training loop is added in **Phase 8c** — this Phase 8a deliverable covers
only the configuration and the data-prep workflow.

---

## Workflow at a glance

```
data/prepare_data.py     →  data/synthetic/{train,test}, data/real/{train,test}
cyclegan/config.py       →  hyperparameters consumed by train.py
cyclegan/train.py        →  Phase 8c — trains G:synthetic→real and F:real→synthetic
cyclegan/test.py         →  Phase 8c — runs the trained generator on test images
scripts/apply_stylization.py → Phase 8e — stylises a full generated dataset
```

---

## 1. Local: prepare the dataset

Run from the project root on your laptop. This step does **not** need a GPU.

```bash
# 1a. Make sure the regular project deps are installed.
source venv/bin/activate
pip install -r requirements.txt

# 1b. Drop your real WB-102 photos into a folder, e.g. ~/Photos/breadboards/
#     (jpg / png / heic all accepted; aim for 100–200 photos covering the
#      lighting / angle / wiring variety described in PHASE_7_8_ARCHITECTURE.md)

# 1c. Build both domains (resized to 256x256, 80/20 train/test split).
python data/prepare_data.py \
    --real-source ~/Photos/breadboards \
    --n-synthetic 500 \
    --image-size 256 \
    --seed 42 \
    --stats-out data/prep_stats.json
```

The script prints a summary like:

```
Domain A — synthetic
  rendered:        500
  train split:     400
  test split:      100
Domain B — real
  found:           150
  train split:     120
  test split:      30
```

If you skip `--real-source`, the synthetic side is built and the `data/real/`
splits stay empty until you supply photos.

---

## 2. Sync the prepared dataset to the GPU host

We're targeting the **NRP (National Research Platform)** Nautilus cluster,
which exposes NVIDIA GPUs through Kubernetes. Adapt the paths to whatever
PVC / scratch volume your namespace has mounted — the steps generalise to
any SSH-accessible GPU host.

```bash
# Option A — direct rsync over SSH to a login node:
rsync -avz --progress \
    data/synthetic data/real \
    user@nrp-login.nautilus.optiputer.net:/scratch/breadboard_cyclegan/data/

# Option B — via kubectl cp into a running pod that has the PVC mounted:
kubectl cp data/synthetic <pod>:/workspace/breadboard/data/
kubectl cp data/real      <pod>:/workspace/breadboard/data/
```

Also push this repo (or at least the `cyclegan/` directory and
`requirements-cyclegan.txt`) so the training scripts are available on the host.

---

## 3. GPU host: install dependencies

PyTorch wheels are CUDA-version specific — pick the build that matches the
driver on the assigned NRP node. Check with `nvidia-smi`. The example below
uses CUDA 12.1.

```bash
# On the GPU pod / node, in a fresh venv or conda env (Python 3.10+):
python -m venv venv-gpu && source venv-gpu/bin/activate

# CUDA-specific torch wheels first:
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# Then the rest of the CycleGAN deps:
pip install -r requirements-cyclegan.txt

# Sanity check.
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The pytorch-CycleGAN-and-pix2pix reference implementation will be vendored
or pip-installed in Phase 8c.

---

## 4. Kick off training (Phase 8c — placeholder command)

The training driver (`cyclegan/train.py`) lands in Phase 8c. The intended
invocation, with this directory's hyperparameter defaults from `config.py`:

```bash
python cyclegan/train.py \
    --data-root /scratch/breadboard_cyclegan/data \
    --checkpoints-dir /scratch/breadboard_cyclegan/checkpoints \
    --name wb102_synth_to_real
```

For a long-running NRP job, wrap that in a Kubernetes Job manifest that
requests one GPU (e.g. `nvidia.com/gpu: 1`) and mounts both the data PVC
and a checkpoint PVC. A reference manifest will be added alongside
`cyclegan/train.py` in Phase 8c.

Defaults (see `cyclegan/config.py`):

| Hyperparameter        | Value     |
|-----------------------|-----------|
| Image size            | 256       |
| Batch size            | 1         |
| Learning rate         | 0.0002    |
| Epochs                | 200       |
| λ cycle               | 10.0      |
| λ identity            | 0.5       |
| Checkpoint interval   | 10 epochs |

Expected wall time: roughly 4–12 hours on a single A10/RTX 3090-class GPU.

---

## 5. Pulling results back

Once training completes, the generator weights end up under
`<checkpoints-dir>/wb102_synth_to_real/`. Sync them back to your laptop and
plug them into `scripts/apply_stylization.py` (Phase 8e) to stylise the
full Phase 6 dataset without touching its labels or bounding boxes.
