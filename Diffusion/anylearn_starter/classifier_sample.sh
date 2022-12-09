root_path=$1
MODEL_FLAGS="--attention_resolutions 32,16,8
            --class_cond True
            --rescale_timesteps True
            --dropout 0.1
            --image_size 64
            --learn_sigma True
            --noise_schedule cosine
            --num_channels 192
            --num_head_channels 64
            --num_res_blocks 3
            --resblock_updown True
            --use_new_attention_order True
            --use_fp16 True
            --use_scale_shift_norm True"

for size in 100 200 300 400 500 600 700 800 900 1000
do
    python classifier_sample.py $MODEL_FLAGS \
      --classifier_scale 1.0 \
      --classifier_path ${root_path}/64x64_classifier.pt \
      --classifier_depth 4 \
      --model_path ${root_path}/64x64_diffusion.pt \
      --save_dir outputs $SAMPLE_FLAGS \
      --diffusion_steps ${size} \
      --class_index -1 \
      --use_ddim False
done


