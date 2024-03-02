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

python test_deeplab_inference_celebahq.py \
--exp experiments/$exp_name \
--resume /mnt/lustre/yslan/Repo/Research/CVPR22_REJ/datasetGAN/datasetGAN/model_dir/face_34_1shot_512_ddf_eg3d/deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_21.pth \
--batch_size 1 --imsize 512 \
--version parsenet --train False \
--test_image_path /mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/CelebAMask-HQ/face_parsing/Data_preprocessing/test_img_crop \
--test_label_path /mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/CelebAMask-HQ/face_parsing/Data_preprocessing/test_label \
# --test_label_path /mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/CelebAMask-HQ/face_parsing/Data_preprocessing/test_label_crop \




# --test_image_path /mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/CelebAMask-HQ/face_parsing/Data_preprocessing/test_img_crop \