
# CUDA_VISIBLE_DEVICES=1 \

# python train_deeplab.py \
# --data_path [path-to-generated-dataset in step4] \
# --exp experiments/<exp_name>.json

# path_generated=dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/face_34/samples/
# exp_name=face_34.json

# march 5th 2022 eccv train 1-shot 
# path_generated=model_dir/face_34_1shot/samples/
# exp_name_512=face_34_1shot.json
# exp_name_128=face_34_1shot_128.json
path_generated=/mnt/lustre/yslan/Dataset/CVPR22/seg
exp_name=face_34_1shot_256_ddf.json
# exp_name=face_34_1shot_256_ddf_allsamples.json

# python train_deeplab.py \
# --data_path $path_generated \
# --exp experiments/$exp_name_128

python train_deeplab.py \
--data_path $path_generated \
--exp experiments/$exp_name \
--ddf_sample \
# --resume model_dir/face_34_1shot_256_ddf/deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_14.pth
