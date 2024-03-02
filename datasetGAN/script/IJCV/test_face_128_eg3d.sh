# path_generated=/mnt/lustre/yslan/Dataset/CVPR22/seg
# exp_name=face_34_1shot_128_ddf.json
exp_name=face_34_1shot_128_ddf_eg3d.json
# model_path=model_dir/face_34_1shot_128_ddf/deeplab_class_34_checkpoint_0_filter_out_0.000000
model_path=model_dir/face_34_1shot_128_ddf_eg3d/deeplab_class_34_checkpoint_0_filter_out_0.000000



python test_deeplab_cross_validation.py --exp experiments/$exp_name \
--resume $model_path  --cross_validate True
