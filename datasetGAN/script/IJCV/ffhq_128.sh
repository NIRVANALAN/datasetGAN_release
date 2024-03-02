exp_name=face_34_1shot_128_ddf_eg3d.json

# Sept 8 1-shot
# path_generated=/mnt/lustre/yslan/Repo/Research/CVPR22_REJ/eccv-ddf-porting/lightning_logs/DatasetSeg_v2/

path_generated=/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out

# 21/April 2023, train the model on eg3d-ddf
python train_deeplab_eg3d.py \
--data_path $path_generated \
--ddf_sample \
--exp experiments/$exp_name

