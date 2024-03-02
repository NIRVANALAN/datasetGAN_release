

# python train_deeplab.py \
# --data_path [path-to-generated-dataset in step4] \
# --exp experiments/<exp_name>.json
path_generated=dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/car_20/samples/
exp_name=car_20.json

CUDA_VISIBLE_DEVICES=3 \
python train_deeplab.py \
--data_path $path_generated \
--exp experiments/$exp_name

# CUDA_VISIBLE_DEVICES=3