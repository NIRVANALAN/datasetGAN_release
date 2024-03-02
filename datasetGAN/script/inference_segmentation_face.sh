model_ckpt=model_dir/face_34/deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_17.pth
# dataset_path='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/inference-tex/FID/eval_results/src'
# output_dir='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/inference-tex/FID/eval_results/seg/src'

#dataset_path='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/inference-tex/FID/eval_results/tgt'
#output_dir='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/inference-tex/FID/eval_results/seg/tgt'


# synthetic -> real
# dataset_path='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/inference-seg/REAL_SEG_MULTIVIEW/eval/tgt'

# for evaluation segmentation
# dataset_path='/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/eccv-ddf-porting/lightning_logs/SEG_METRICS_GT/seed99/eval_results/tgt'

# for highres G T
# dataset_path=/mnt/lustre/yslan/Repo/Research/SIGA22/BaseModels/baseline_stylesdf/evaluations/mean/ffhq1024x1024/final_model/fixed_angles/images

# celebamask G
dataset_path=/mnt/lustre/yslan/Repo/Research/SIGA22/BaseModels/baseline_stylesdf/evaluations/celebamask/ffhq1024x1024/final_model/fixed_angles/images


# ~/repo/3d/correspondence/pigan_densecorr/out/deform/v0.99/size999_seed1_celeba_z_8w/template.png

# March 9th 2023, ddf eg3d version
#dataset_path='/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/template'
#output_dir='/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/seg'

# March 9th 2023, DDF eg3d segmetnation evaluation set
dataset_path='/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/segmentation'
output_dir='/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/segmentation_for_metrics'

python demo.py --ckpt $model_ckpt --deeplab_res 512 \
--dataset_path $dataset_path \
# --output_path $output_dir
