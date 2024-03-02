#!/bin/bash
export PYTHONPATH=$PWD

resume_dir=dataset_release/datasetgan_release_checkpoints/interpreter_checkpoint/face_34

# CUDA_VISIBLE_DEVICES=0  python train_interpreter.py  --exp experiments/face_34.json --resume $resume_dir --num_sample  2500 --start_step 0 --generate_data True &
# CUDA_VISIBLE_DEVICES=1  python train_interpreter.py  --exp experiments/face_34.json --resume $resume_dir --num_sample  2500 --start_step 2500 --generate_data True &
# CUDA_VISIBLE_DEVICES=2  python train_interpreter.py  --exp experiments/face_34.json --resume $resume_dir --num_sample  2500 --start_step 5000 --generate_data True &
# CUDA_VISIBLE_DEVICES=3  python train_interpreter.py  --exp experiments/face_34.json --resume $resume_dir --num_sample  2500 --start_step 7500 --generate_data True

# python train_interpreter.py --generate_data True --exp experiments/face_34_1shot.json --resume model_dir/face_34_1shot/ --num_sample 10000
CUDA_VISIBLE_DEVICES=2  python train_interpreter.py  --exp experiments/face_34_1shot.json --resume $resume_dir --num_sample  3334 --start_step 0 --generate_data True &
CUDA_VISIBLE_DEVICES=3  python train_interpreter.py  --exp experiments/face_34_1shot.json --resume $resume_dir --num_sample  3334 --start_step 3334 --generate_data True &
CUDA_VISIBLE_DEVICES=4  python train_interpreter.py  --exp experiments/face_34_1shot.json --resume $resume_dir --num_sample  3334 --start_step 6668 --generate_data True
