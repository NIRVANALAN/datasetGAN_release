exp_name=face_34_1shot_256_ddf_ffhq.json

# Sept 8 1-shot
# path_generated=/mnt/lustre/yslan/Repo/Research/CVPR22_REJ/eccv-ddf-porting/lightning_logs/DatasetSeg_v2/



# Sept 22, ffhq 1-shot, little_endian 2, 9.5K images baselines, 256 res
path_generated=/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/cvpr22_pigan_dc_pl/project/lightning_logs/sdf/ffhq/inference/feat_1_littleendian/

python train_deeplab.py \
--data_path $path_generated \
--ddf_sample \
--exp experiments/$exp_name

