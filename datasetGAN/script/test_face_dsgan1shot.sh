# dsgan 1shot
exp_name=face_34_1shot_256.json
model_path=model_dir/face_34_1shot_256/deeplab_class_34_checkpoint_0_filter_out_0.000000
# exp_name=face_34_1shot_128.json
# model_path=model_dir/face_34_1shot_128/deeplab_class_34_checkpoint_0_filter_out_0.000000


python test_deeplab_cross_validation.py --exp experiments/$exp_name \
--resume $model_path  --cross_validate False
