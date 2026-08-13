# BraTS-METS 2026 Task 1 submission: M_base (R37b, 5-fold nnU-Net) + M_FPR (ET
# false-positive rejector). Reproduces challenge submission 9771640 (ET 0.7187 /
# mean 0.6660). Zero network access at runtime -- everything below is baked in
# at build time.
#
# Platform: NVIDIA A10G (24GB VRAM), 16 vCPU, 200GB storage, max CUDA 13.0.
# torch 2.12.0+cu130 (pinned to match the training environment) already satisfies
# the CUDA 13.0 ceiling.
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app

# No apt packages needed: nibabel/scipy.ndimage/skimage.measure/sklearn/nnunetv2
# are all pure-Python + compiled-wheel dependencies with no OpenGL/GUI requirement,
# so we deliberately skip apt-get here (also sidesteps rootless-container dbus/
# systemd issues with `apt-get` on some Docker/podman hosts).

# One COPY per artifact group for layer caching (per the challenge's linked
# sample-model-templates convention).
COPY requirements.txt .
# --break-system-packages: this base image's Python is Debian/PEP-668 managed;
# safe here since the container has no other purpose than running this pipeline.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Bake the custom trainer into nnU-Net's own package tree -- nnU-Net's predict CLI
# must be able to import it by name when reconstructing the network architecture
# from the checkpoint (mirrors the `cp ... $VARIANTS_DIR/` step every training/
# predict SLURM script in this project performs).
RUN python -c "import nnunetv2, os, shutil; \
    d = os.path.join(os.path.dirname(nnunetv2.__file__), 'training/nnUNetTrainer/variants/loss'); \
    print(d)" > /tmp/variants_dir.txt
COPY trainers/nnUNetTrainerDiceTopK10Loss_5000epochs.py /tmp/trainer.py
RUN VARIANTS_DIR=$(cat /tmp/variants_dir.txt) && cp /tmp/trainer.py "$VARIANTS_DIR/" \
    && python -c "\
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; \
import nnunetv2.training.nnUNetTrainer.variants as v; \
c = recursive_find_python_class(v.__path__[0], 'nnUNetTrainerDiceTopK10Loss_5000epochs', 'nnunetv2.training.nnUNetTrainer.variants'); \
assert c is not None, 'custom trainer not discoverable after build-time install'; \
print('trainer ok:', c)"

# Vendored model weights + plans + rejector (~1.2GB) and the inference scripts.
COPY model/ /app/model/
COPY src/ /app/scripts/

# /input is mounted read-only, /output read-write by the platform at run time;
# these just ensure the mountpoints exist for local testing outside that harness.
RUN mkdir -p /input /output /work

ENTRYPOINT ["python", "/app/scripts/run_inference.py"]
