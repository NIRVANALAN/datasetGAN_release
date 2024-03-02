cat=cat_16
face=face_34
car=car_20

# checkpoint=model_dir/$cat/deeplab_class_16.000000 
checkpoint=model_dir/$cat/deeplab_class_16_checkpoint_0_filter_out_0.000000
python test_deeplab_cross_validation.py --exp experiments/$cat.json \
--resume $checkpoint \
--cross_validate True
# [path-to-downstream task checkpoint] 

# Face 17
# Cat 12
# Car

