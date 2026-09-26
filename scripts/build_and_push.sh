#!/usr/bin/env bash
# Build the training image and push it to a Quay registry.
#
# Run from the repository root, on the build machine:
#     export REGISTRY='<registry host>' REGISTRY_ORG='<organisation>' QUAY_USER='<robot account>'
#     export QUAY_TOKEN='...'        # never commit this, never paste it in chat
#     ./scripts/build_and_push.sh v001
#
set -euo pipefail

REGISTRY="${REGISTRY:?set REGISTRY to the registry host}"
ORG="${REGISTRY_ORG:?set REGISTRY_ORG to the registry organisation}"
IMAGE="qwen35-ocr-train"
USERNAME="${QUAY_USER:?set QUAY_USER to the registry robot account}"
TAG="${1:-}"

if [[ -z "$TAG" ]]; then
    echo "usage: $0 <tag>      e.g. $0 v001" >&2
    echo "never use 'latest' for an experiment that matters" >&2
    exit 1
fi
if [[ "$TAG" == "latest" ]]; then
    echo "ERROR: refusing 'latest'. Use a versioned tag (v001, v002, ...)." >&2
    exit 1
fi
if [[ -z "${QUAY_TOKEN:-}" ]]; then
    echo "ERROR: QUAY_TOKEN is not set." >&2
    echo "  export QUAY_TOKEN='<token>'   # from a password manager, not from chat" >&2
    exit 1
fi
if [[ -z "${BASE_IMAGE:-}" ]]; then
    echo "ERROR: BASE_IMAGE is not set. Find the one this cluster already runs:" >&2
    echo "  oc get deployment <A_RUNNING_GPU_DEPLOYMENT> \\" >&2
    echo "    -o jsonpath='{.spec.template.spec.containers[0].image}'" >&2
    exit 1
fi

REF="${REGISTRY}/${ORG}/${IMAGE}:${TAG}"

echo "==> disk check"
# Build layers for a CUDA/PyTorch image run to tens of GB. VM01 ships with
# ~10 GB free, which is not enough; the guide asks for a +50 GB extension.
AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "    ${AVAIL}G available on /"
if (( AVAIL < 40 )); then
    echo "ERROR: need ~40G+ free to build. Ask infra for the VM01 disk extension." >&2
    exit 1
fi

echo "==> refusing to build if a secret leaked into the build context"
# Generic shapes, not literal tokens: a 40+ char uppercase/digit blob
# (Quay robot tokens look like that), or a secret-ish variable assigned a value.
if grep -rInE '[A-Z0-9]{40,}|(SECRET|TOKEN|PASSWORD|ACCESS_KEY) *= *"'"'"'[A-Za-z0-9]' \
       --exclude-dir=.git --exclude='build_and_push.sh' . 2>/dev/null; then
    echo "ERROR: credential-shaped string in the build context. Remove it." >&2
    exit 1
fi

echo "==> build  ${REF}"
podman build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${REF}" -f Containerfile .

echo "==> verify imports inside the freshly built image"
podman run --rm "${REF}" -lc 'python /opt/ocr-training/scripts/preflight.py --imports-only'

echo "==> login"
# --password-stdin keeps the token out of the process list and shell history.
printf '%s' "${QUAY_TOKEN}" | \
    podman login "${REGISTRY}" -u "${USERNAME}" --password-stdin --tls-verify=false

echo "==> push"
podman push "${REF}" --tls-verify=false

echo "==> digest (record this in the experiment metadata)"
podman inspect --format '{{index .RepoDigests 0}}' "${REF}" || true

echo "==> done: ${REF}"
