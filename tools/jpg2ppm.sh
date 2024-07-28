PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/jpg2ppm.py --jpg_root ./tools/data/nuscenes/samples/ \
       --ppm_root ./tools/data/ld_nuscenes/samples/ \
