# Training image for Qwen3.5-9B OCR fine-tuning.
#
# Contains SOFTWARE ONLY. Deliberately absent:
#   - model weights   (staged to the PVC from MinIO at run time)
#   - the dataset     (staged to the PVC from MinIO at run time)
#   - credentials     (injected from an OpenShift Secret at run time)
#
# The cluster is air-gapped, so nothing here may reach the Internet at runtime.

# ---------------------------------------------------------------------------
# BASE IMAGE — must be an image the cluster already runs successfully.
# Do not use a public docker.io image: the cluster cannot pull it.
# Discover the right one from a working GPU deployment:
#
#   oc get deployment <A_RUNNING_GPU_DEPLOYMENT> \
#     -o jsonpath='{.spec.template.spec.containers[0].image}'
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=REPLACE_ME_WITH_INTERNAL_PYTORCH_CUDA_IMAGE
FROM ${BASE_IMAGE}

USER root
WORKDIR /opt/ocr-training

# Dependencies first, so edits to our own code do not invalidate this layer.
COPY requirements.txt /opt/ocr-training/
RUN python -m pip install --no-cache-dir -r /opt/ocr-training/requirements.txt

COPY scripts/ /opt/ocr-training/scripts/
COPY src/     /opt/ocr-training/src/
COPY configs/ /opt/ocr-training/configs/

ENV PYTHONPATH=/opt/ocr-training \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    TORCH_HOME=/workspace/cache/torch \
    TMPDIR=/workspace/tmp \
    PYTHONUNBUFFERED=1

# OpenShift runs containers as an arbitrary non-root UID in the root group.
# Anything the process writes to must therefore be group-writable.
RUN chmod -R g=u /opt/ocr-training && \
    mkdir -p /workspace && chmod -R g=u /workspace

ENTRYPOINT ["/bin/bash"]
