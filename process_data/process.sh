export OPENBLAS_NUM_THREADS=1

# Please install https://github.com/autonomousvision/navsim.
# This is used for generating High-Level Driving Commands, used in E2E AD.
export NAVSIM_DEVKIT_ROOT=/PATH_TO/navsim
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}

split=trainval
# Please download all the nuplan data from https://www.nuscenes.org/nuplan.
export NUPLAN_PATH=/PATH_TO/nuplan_root/dataset/nuplan-v1.1
export NUPLAN_DB_PATH=${NUPLAN_PATH}/splits/${split}
export NUPLAN_SENSOR_PATH=/PATH_TO/nuplan_root/dataset/nuplan-v1.1/sensor
export NUPLAN_MAPS_ROOT=/PATH_TO/nuplan_root/dataset/nuplan-maps-v1.0

OUT_DIR='$HOME/rap_workspace/dataset/navsim_logs/trainval'
# 1. Generate OpenScene metadata and 3D rasterized multi-camera views 
#    for all nuPlan logs (~1200h).
python -u create_openscene_metadata.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561

OUT_DIR='$HOME/rap_workspace/dataset_perturbed/navsim_logs/trainval'
# 2. Generate recovery-oriented trajectory perturbations 
python -u create_openscene_metadata_perturbed.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561

OUT_DIR='$HOME/rap_workspace/dataset_aug/navsim_logs/trainval'
# 3. Generate cross-agent view synthesis 
python -u create_openscene_metadata_aug.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561