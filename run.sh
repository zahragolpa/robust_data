transformations=("blur" "shot_noise" "rotate" "random_shadow" "sharpen_and_darken" "random_grid_shuffle"
                                  "horizontal_flip" "vertical_flip" "horizontal_vertical_flip" "sun_flare"
                                  "contrast_raise" "brightness_raise" "brightness_reduce" "red_shift" "green_shift"
                                  "blue_shift" "yellow_shift" "magenta_shift")
#transformations=("cyan_shift" "translate_horizontal" "translate_vertical"
#                                  "translate_horizontal_reflect" "translate_vertical_reflect" "gaussian_noise_2"  "impulse_noise"
#                                  "defocus_blur" "motion_blur_2" "snow" "frost" "fog" "elastic_transform"
#                                  "pixelate" "jpeg_compression" "zoom_center"
#                                  "grid_distortion_repeat_edge" "grid_distortion_resize" "rain_drizzle_slant_10" "convex_hull_black" "convex_hull_white"
#                                 "edge" "inverted_edge")
#

for t in ${transformations[@]}
do
   accelerate launch data_statistics.py --epochs 10 --bsz 64 -t $t
   cd plot_utils
   python plotting.py --exp "cifar10_epochs_10_lr_2e-05_bsz_64_t_"$t"_dynamic"
   cd ../

done
