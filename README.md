<h1 align="center">
  Conditional Balance:  <br>
  Improving Multi-Conditioning Trade-Offs in Image Generation <br>
</h1>

This is the official repository of the paper "Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation" by Nadav Z. Cohen, Oron Nir, and Ariel Shamir.

![teaser](assets/results_text.png)

# Inference

### Text Conditioned
The following call can be used to generate text-conditioned images:
```
python style_gen_img.py --seed 10 \
                        --content_prompts "A tabby cat sitting" \
                        --style_prompt "in the style of Vincent van-Gogh" \
                        --reference_prompt "An oil painting" \
                        --lambda_s 0.43 \
                        --num_images_per_prompt 1 \
                        --output_path "outputs" \
                        --initialize_latents
```

### Multi-Text Conditioned
You can also generate text-conditioned images using more than one text condition:
```
python style_gen_img.py --seed 10 \
                        --content_prompts "A persian cat sitting" "A St. Bernard dog running" "A black bear resting on the beach" \
                        --style_prompt "in the style of Vincent van-Gogh" \
                        --reference_prompt "An oil painting" \
                        --lambda_s 0.43 \
                        --num_images_per_prompt 1 \
                        --output_path "outputs" \
                        --initialize_latents
```

### ControlNet Conditioned
The following call can be used to generate canny conditioned images. Use the same parameters with "style_gen_img_depth.py" for depth conditioning.
```
python style_gen_img_canny.py --seed 55 \
                              --content_prompts "A panda bear standing on a boulder in the forest" \
                              --style_prompt "in the style of Henri Matisse" \
                              --reference_prompt "A painting" \
                              --content_image_path "assets/panda.jpg" \
                              --lambda_s 0.57 \
                              --lambda_t 0.8 \
                              --num_images_per_prompt 3 \
                              --output_path "outputs" \
                              --initialize_latents
```

### Multi-ControlNet Conditioned
You can also generate canny conditioned images using multiple cannies. To do this, <content_image_path> should be a path of a directory with the conditioning images. <br>
In this setting, if you use a single content prompt it will be applied to all canny conditionals, otherwise you have to input a prompt to each canny in the inputs folder.
```
python style_gen_img_canny.py --seed 55 \
                              --content_prompts "A bernedoodle sitting on grass" "A marble scalpture of David" "A Galapagos tortoise "\
                              --style_prompt "in the style of Henri Matisse" \
                              --reference_prompt "A painting" \
                              --content_image_path "assets/conditionals_dir_example" \
                              --lambda_s 0.57 \
                              --lambda_t 0.8 \
                              --num_images_per_prompt 3 \
                              --output_path "outputs" \
                              --initialize_latents
```



# Citation
If you found this project helpful in your research, please consider citing our paper.
```
@misc{cohen2024conditionalbalanceimprovingmulticonditioning,
      title={Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation}, 
      author={Nadav Z. Cohen and Oron Nir and Ariel Shamir},
      year={2024},
      eprint={2412.19853},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.19853}, 
}
```