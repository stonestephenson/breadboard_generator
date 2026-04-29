# Jupyter GPU — Common Commands

Reference for running the CycleGAN pipeline on the Jupyter GPU box.
All commands assume you're at `~/breadboard_generator` with the venv active.

```bash
cd ~/breadboard_generator
source venv/bin/activate
```

---

## 1. Pulling Latest Code

```bash
git pull origin main
```

---

## 2. Sanity Checks

```bash
# Confirm GPU is visible to PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Confirm HEIC support is registered
python -c "from pillow_heif import register_heif_opener; register_heif_opener(); from PIL import Image; print('ok')"

# Count raw photos
ls data/real/*.HEIC | wc -l
```

---

## 3. Resetting Data State

### Wipe prepared splits, keep raw HEICs
```bash
rm -rf data/real/train data/real/test data/synthetic data/prep_stats.json data/real_cropped
```

### Wipe raw HEICs too (full reset before re-uploading)
```bash
rm -f data/real/*.HEIC data/real/*.heic
rm -rf data/real/train data/real/test data/synthetic data/prep_stats.json data/real_cropped
```

### Wipe CycleGAN training artifacts
```bash
rm -rf cyclegan/checkpoints cyclegan/samples cyclegan/logs
```

---

## 4. Data Preparation

```bash
python data/prepare_data.py \
    --real-source data/real \
    --clean \
    --stats-out data/prep_stats.json
```

After running, sanity-check:
```bash
cat data/prep_stats.json
ls data/real/train | wc -l
ls data/real/test  | wc -l
ls data/synthetic/train | wc -l
```

---

## 5. CycleGAN Training

### Start fresh (recommended: batch_size 4 for RTX A6000)
```bash
nohup python -m cyclegan.train --batch-size 4 > train.log 2>&1 &
echo $! > train.pid
echo "Training PID: $(cat train.pid)"
tail -f train.log
```

### Resume from latest checkpoint
```bash
ls -lt cyclegan/checkpoints/                       # find the latest epoch_NNN.pth

nohup python -m cyclegan.train \
    --batch-size 4 \
    --resume cyclegan/checkpoints/epoch_NNN.pth \
    > train.log 2>&1 &
echo $! > train.pid
tail -f train.log
```

`Ctrl-C` only stops the `tail`, not the training — `nohup` keeps the
training process alive across disconnects.

---

## 6. Monitoring Training

```bash
# Is training still alive?
ps -p $(cat train.pid)

# Latest training output
tail -50 train.log

# Last error (if it crashed)
tail -100 train.log

# Latest checkpoint
ls -lt cyclegan/checkpoints/ | head

# Latest sample translation images
ls -lt cyclegan/samples/ | head

# GPU utilization (run in separate terminal)
nvidia-smi
```

What healthy training looks like in `nvidia-smi`:
- GPU utilization 90–100%
- Power draw near max (~280W on A6000)
- Your training PID listed under "Processes"

---

## 7. Stopping Training

```bash
kill $(cat train.pid)
```

If it doesn't die in a few seconds:
```bash
kill -9 $(cat train.pid)
```

---

## 8. Full Reset and Restart (Nuclear Option)

```bash
cd ~/breadboard_generator
source venv/bin/activate

# Stop any running training
kill $(cat train.pid) 2>/dev/null

# Wipe everything regenerable
rm -rf data/real/train data/real/test data/synthetic data/prep_stats.json data/real_cropped
rm -rf cyclegan/checkpoints cyclegan/samples cyclegan/logs

# Pull latest code
git pull origin main

# Re-prep
python data/prepare_data.py --real-source data/real --clean --stats-out data/prep_stats.json

# Re-train
nohup python -m cyclegan.train --batch-size 4 > train.log 2>&1 &
echo $! > train.pid
tail -f train.log
```

---

## Notes

- **Checkpoints save every 10 epochs** to `cyclegan/checkpoints/epoch_NNN.pth`.
- **Sample translations save every 500 iterations** to `cyclegan/samples/`.
- **CSV loss log appends** — old data stays unless you wipe `cyclegan/logs/`.
- **Default training is 200 epochs** (~2 hours at batch_size 4 on A6000).
- **LR decay starts at epoch 100** — losses should keep slowly improving but stabilize in second half.
