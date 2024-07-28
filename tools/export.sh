
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/export.py ./tools/configs/stage2_e2e/base_e2e.py ./tools/ckpts/uniad_base_e2e.pth --filepath=./tools/ckpts/uniad_base_e2e.bin