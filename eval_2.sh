python infz.py \
    --root_dir /home/jovyan/workspace/bagel/MathVista/test \
    --checkpoint_dir /home/jovyan/workspace/bagel/h200-ckpt-0000650 \
    --checkpoint_file model_bf16.safetensors \
    --base_model_path /dev/shm/models/BAGEL-7B-MoT \
    --device_mem 80GiB \
    --gpu_id 5