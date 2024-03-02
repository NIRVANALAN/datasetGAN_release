exp_name=face_34_1shot_512_ddf_eg3d.json

# Sept 8 1-shot
# path_generated=/mnt/lustre/yslan/Repo/Research/CVPR22_REJ/eccv-ddf-porting/lightning_logs/DatasetSeg_v2/

path_generated=/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out

# # 21/April 2023, train the model on eg3d-ddf
# python train_deeplab_eg3d.py \
# --data_path $path_generated \
# --ddf_sample \
# --exp experiments/$exp_name


# 23/April 2023, inference fail, check the performance of training time
python train_deeplab_eg3d.py \
--data_path $path_generated \
--ddf_sample \
--exp experiments/$exp_name \
--resume /mnt/lustre/yslan/Repo/Research/CVPR22_REJ/datasetGAN/datasetGAN/model_dir/face_34_1shot_512_ddf_eg3d/deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_18.pth \
# --resume /mnt/lustre/yslan/Repo/Research/CVPR22_REJ/datasetGAN/datasetGAN/model_dir/face_34_1shot_512_ddf_eg3d/old_deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_19.pth \
