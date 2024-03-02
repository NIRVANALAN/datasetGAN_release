# json_name=face_34_1shot_256_ddf_ffhq
# json_name=face_34_1shot_256_ddf
json_name=face_34_1shot_256

# no need to change code below
exp_name=$json_name.json
model_path=model_dir/$json_name/deeplab_class_34_checkpoint_0_filter_out_0.000000

python test_deeplab_cross_validation.py --exp experiments/$exp_name \
--resume $model_path  \
--cross_validate True
