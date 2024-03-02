 python train_interpreter.py \
   --exp experiments/face_34.json --resume \
 dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/face_34 \
 --num_sample  5 --start_step 0 --generate_data True \
 --save_vis False

 python train_interpreter.py \
   --exp experiments/cat_16.json --resume \
 dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/cat_16 \
 --num_sample  5 --start_step 0 --generate_data True # --save_vis False

 python train_interpreter.py \
   --exp experiments/cat_16.json --resume \
 dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/cat_16 \
 --num_sample  5 --start_step 0 --generate_data True --save_vis True


#python train_interpreter.py \
#  --exp experiments/face_34.json --resume \
#dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/face_34 \
#--num_sample  5 --start_step 0 --inference --img_path \
#~/repo/3d/correspondence/pigan_densecorr/out/deform/v0.99/size999_seed1_celeba_z_8w/template.png
