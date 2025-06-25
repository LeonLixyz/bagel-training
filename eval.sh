python get_math_vista.py \
    --output_root /home/jovyan/workspace/bagel/MathVista-1000 \
    --splits testmini

for rank in {0..7}; do
    python infz.py \
        --root_dir /home/jovyan/workspace/bagel/MathVista-1000/testmini \
        --checkpoint_dir /home/jovyan/workspace/bagel/h200-ckpt-0001000 \
        --checkpoint_file model_bf16.safetensors \
        --base_model_path /dev/shm/models/BAGEL-7B-MoT \
        --device_mem 80GiB \
        --rank $rank \
        --world_size 8 &
done
wait