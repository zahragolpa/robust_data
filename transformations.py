import albumentations as A
import os
from matplotlib import pyplot as plt
import cv2
from albumentations.augmentations.functional import _maybe_process_in_chunks
from imagecorruptions import corrupt
import math
import numpy as np
from random import uniform


list_of_transformations = ['rotate', 'blur', 'random_shadow', 'sharpen_and_darken', 'random_grid_shuffle',
        'gaussian_noise', 'motion_blur', 'horizontal_flip', 'vertical_flip', 'horizontal_vertical_flip', 'sun_flare',
        'contrast_raise', 'brightness_raise', 'brightness_reduce', 'red_shift', 'green_shift',
        'blue_shift', 'yellow_shift', 'magenta_shift', 'cyan_shift', 'translate_horizontal', 'translate_vertical',
        'translate_horizontal_reflect', 'translate_vertical_reflect', 'gaussian_noise_2', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'motion_blur_2', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
        'pixelate', 'jpeg_compression', 'zoom_center', 'zoom_right', 'zoom_left', 'zoom_top', 'zoom_bottom',
        'zoom_right_bottom', 'zoom_right_top', 'zoom_left_bottom', 'zoom_left_top',
        'grid_distortion_repeat_edge', 'grid_distortion_resize', 'rain_drizzle_slant_0', 'rain_drizzle_slant_10',
        'rain_drizzle_slant_neg_10', 'rain_drizzle_slant_20', 'rain_drizzle_slant_neg_20', 'rain_heavy_slant_0',
        'rain_heavy_slant_10', 'rain_heavy_slant_neg_10', 'rain_heavy_slant_20', 'rain_heavy_slant_neg_20',
        'rain_torrential_slant_0', 'rain_torrential_slant_10', 'rain_torrential_slant_neg_10',
        'rain_torrential_slant_20', 'rain_torrential_slant_neg_20', 'convex_hull_black', 'convex_hull_white',
        'edge', 'inverted_edge', 'fisheye', 'pincushion',
        'camera_shake_upper_left', 'camera_shake_up','camera_shake_upper_right', 'camera_shake_right',
        'camera_shake_lower_right', 'camera_shake_down', 'camera_shake_lower_left', 'camera_shake_left',
        'camera_shake_all_directions', 'gamma_bright', 'gamma_dark']

range_dict = {
    'rotate': (0, 360), 'blur': (3, 30), 'random_shadow': (0, 1), 'sharpen_and_darken': (0, 1), 'random_grid_shuffle': (2, 5),
    'gaussian_noise': (0, 1000), 'motion_blur': (3, 30), 'horizontal_flip': (0, 1), 'vertical_flip': (0, 1), 'horizontal_vertical_flip': (0, 1),
    'sun_flare': (75, 150), 'contrast_raise': (0.2, 1), 'brightness_raise': (0.2, 1), 'brightness_reduce': (0.2, 1),
    'red_shift': (1, 127), 'green_shift': (1, 127),
    'blue_shift': (1, 127), 'yellow_shift': (1, 127), 'magenta_shift': (1, 127), 'cyan_shift': (1, 127),
    'translate_horizontal': (0.2, 0.5), 'translate_vertical': (0.2, 0.5),
    'translate_horizontal_reflect': (0.2, 0.8), 'translate_vertical_reflect': (0.2, 0.8), 'gaussian_noise_2': (1, 5),
    'shot_noise': (1, 5), 'impulse_noise': (1, 5)
    # 'defocus_blur', 'motion_blur_2', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
    # 'pixelate', 'jpeg_compression', 'zoom_center', 'zoom_right', 'zoom_left', 'zoom_top', 'zoom_bottom',
    # 'zoom_right_bottom', 'zoom_right_top', 'zoom_left_bottom', 'zoom_left_top',
    # 'grid_distortion_repeat_edge', 'grid_distortion_resize', 'rain_drizzle_slant_0', 'rain_drizzle_slant_10',
    # 'rain_drizzle_slant_neg_10', 'rain_drizzle_slant_20', 'rain_drizzle_slant_neg_20', 'rain_heavy_slant_0',
    # 'rain_heavy_slant_10', 'rain_heavy_slant_neg_10', 'rain_heavy_slant_20', 'rain_heavy_slant_neg_20',
    # 'rain_torrential_slant_0', 'rain_torrential_slant_10', 'rain_torrential_slant_neg_10',
    # 'rain_torrential_slant_20', 'rain_torrential_slant_neg_20', 'convex_hull_black', 'convex_hull_white',
    # 'edge', 'inverted_edge', 'fisheye', 'pincushion',
    # 'camera_shake_upper_left', 'camera_shake_up','camera_shake_upper_right', 'camera_shake_right',
    # 'camera_shake_lower_right', 'camera_shake_down', 'camera_shake_lower_left', 'camera_shake_left',
    # 'camera_shake_all_directions', 'gamma_bright', 'gamma_dark'

}


def get_t_range(t_name):
    return (range_dict[t_name][0], range_dict[t_name][1]/2)

small_count = 0

def get_local_path(path):
    this_dir = os.path.dirname(__file__)
    return os.path.join(os.path.abspath(this_dir), path)

def rotate(input_image_numpy, parameter=45):
    test_name = "rotate"
    description = 'https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate'
    parameter = int(round(parameter))
    transform = A.Compose([A.Rotate(limit=[parameter, parameter], interpolation=1, border_mode=0, value=0, mask_value=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def translate_horizontal(input_image_numpy, parameter=0.1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[0, 0], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[parameter, parameter], shift_limit_y=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def translate_vertical(input_image_numpy, parameter=0.1):
    test_name = "translate_vertical"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[0, 0], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=None, shift_limit_y=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def translate_horizontal_reflect(input_image_numpy, parameter=0.3):
    test_name = "translate_horizontal_reflect"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[0, 0], rotate_limit=[0, 0], interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=[parameter, parameter], shift_limit_y=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def translate_vertical_reflect(input_image_numpy, parameter=0.3):
    test_name = "translate_vertical_reflect"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate (shift_limit=[0, 0], scale_limit=[0, 0], rotate_limit=[0, 0], interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_center(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_right(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[-parameter/2.0, -parameter/2.0], shift_limit_y=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_left(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[parameter/2.0, parameter/2.0], shift_limit_y=None, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_top(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=None, shift_limit_y=[parameter/2.0, parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_bottom(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=None, shift_limit_y=[-parameter/2.0, -parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_right_bottom(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[-parameter/2.0, -parameter/2.0], shift_limit_y=[-parameter/2.0, -parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_right_top(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[-parameter/2.0, -parameter/2.0], shift_limit_y=[parameter/2.0, parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_left_bottom(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[parameter/2.0, parameter/2.0], shift_limit_y=[-parameter/2.0, -parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def zoom_left_top(input_image_numpy, parameter=1):
    test_name = "translate_horizontal"
    description = ''
    transform = A.Compose([A.ShiftScaleRotate(shift_limit=[0, 0], scale_limit=[parameter, parameter], rotate_limit=[0, 0], interpolation=1, border_mode=0, value=0, mask_value=None, shift_limit_x=[parameter/2.0, parameter/2.0], shift_limit_y=[parameter/2.0, parameter/2.0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def blur(input_image_numpy, parameter=15):
    test_name = "blur"
    description = 'https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur'
    parameter = int(parameter)
    if parameter < 3:
        parameter = 3
    transform = A.Compose([A.Blur([parameter, parameter], p=1)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def one_image_solarize(input_image_numpy, parameter):
    test_name = "solarize"
    description = ''
    transform = A.Compose([A.Solarize(parameter, p=1)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def random_shadow(input_image_numpy, parameter=1):
    test_name = "random_shadow"
    description = ''
    if parameter < 0:
        parameter = 0
    if parameter > 1:
        parameter = 1
    transform = A.Compose([A.RandomShadow(shadow_roi=(0.5-(parameter/2), 0.5-(parameter/2), 0.5+(parameter/2), 0.5+(parameter/2)), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def sharpen_and_darken(input_image_numpy, parameter=0.5):
    test_name = "sharpen_and_darken"
    description = ''
    transform = A.Compose([A.Sharpen(alpha=[0.5, 0.5], lightness=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def random_grid_shuffle(input_image_numpy, parameter=2):
    test_name = "random_grid_shuffle"
    description = ''
    if parameter < 0:
        parameter = 1
    transform = A.Compose([A.RandomGridShuffle(grid=[int(round(parameter)), int(round(parameter))], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def gaussian_noise(input_image_numpy, parameter=2000):
    test_name = "gaussian_noise"
    description = ''
    transform = A.Compose([A.GaussNoise(var_limit=[int(round(parameter)), int(round(parameter))], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def motion_blur(input_image_numpy, parameter=16.5):
    test_name = "motion_blur"
    description = ''
    if parameter < 3:
        parameter = 3
    transform = A.Compose([A.MotionBlur(blur_limit=[int(round(parameter)), int(round(parameter))], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def horizontal_flip(input_image_numpy, parameter=1):
    test_name = "horizontal_flip"
    description = ''
    parameter = 1
    transform = A.Compose([A.HorizontalFlip(always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def vertical_flip(input_image_numpy, parameter=1):
    test_name = "vertical_flip"
    description = ''
    parameter = 1
    transform = A.Compose([A.VerticalFlip(always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def horizontal_vertical_flip(input_image_numpy, parameter=1):
    test_name = "vertical_flip"
    description = ''
    parameter = 1
    transform = A.Compose([
        A.HorizontalFlip(always_apply=True),
        A.VerticalFlip(always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def sun_flare(input_image_numpy, parameter=150):
    test_name = "sun_flare"
    description = ''
    if parameter < 3:
        parameter = 3
    transform = A.Compose([A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0,
                              angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=7, src_radius=1+int(round(parameter)),
                              src_color=(255, 255, 255), always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def contrast_raise(input_image_numpy, parameter=0.45):
    test_name = "contrast_raise"
    description = ''
    transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=[0, 0], contrast_limit=[parameter, parameter], brightness_by_max=True, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def brightness_raise(input_image_numpy, parameter=0.3):
    test_name = "brightness_raise"
    description = ''
    transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=[parameter, parameter], contrast_limit=[0, 0], brightness_by_max=True, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def brightness_reduce(input_image_numpy, parameter=0.3):
    test_name = "brightness_reduce"
    description = ''
    transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=[-parameter, -parameter], contrast_limit=[0, 0], brightness_by_max=True, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def desert_domain_adaptation(input_image_numpy, parameter):
    test_name = "desert_domain_adaptation"
    description = 'Desert_domain_adaptation description'
    im = plt.imread(get_local_path(r'domain_adaptation/desert.jpg'))[:, :, :3]
    transform = A.Compose(
        [A.HistogramMatching([im], blend_ratio=(parameter, parameter), read_fn=lambda x: x, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def winter_domain_adaptation(input_image_numpy, parameter):
    test_name = "winter_domain_adaptation"
    description = ''
    im = plt.imread(get_local_path(r'domain_adaptation/winter.jpg'))[:, :, :3]
    transform = A.Compose(
        [A.HistogramMatching([im], blend_ratio=(parameter, parameter), read_fn=lambda x: x, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def jungle_domain_adaptation(input_image_numpy, parameter):
    test_name = "jungle_domain_adaptation"
    description = ''
    im = plt.imread(get_local_path(r'domain_adaptation/jungle.jpg'))[:, :, :3]
    transform = A.Compose(
        [A.HistogramMatching([im], blend_ratio=(parameter, parameter), read_fn=lambda x: x, always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def red_shift(input_image_numpy, parameter=50):
    test_name = "red_shift"
    description = 'https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift'
    # print(input_image_numpy)
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[parameter, parameter], g_shift_limit=[0, 0],
                                      b_shift_limit=[0, 0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    # print(transformed_image)
    return transformed_image

def green_shift(input_image_numpy, parameter=50):
    test_name = "green_shift"
    description = ''
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[0, 0], g_shift_limit=[parameter, parameter],
                                      b_shift_limit=[0, 0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def blue_shift(input_image_numpy, parameter=50):
    test_name = "blue_shift"
    description = ''
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[0, 0], g_shift_limit=[0, 0],
                                      b_shift_limit=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def yellow_shift(input_image_numpy, parameter=50):
    test_name = "yellow_shift"
    description = ''
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[parameter, parameter], g_shift_limit=[parameter, parameter],
                                      b_shift_limit=[0, 0], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def magenta_shift(input_image_numpy, parameter=50):
    test_name = "magenta_shift"
    description = ''
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[parameter, parameter], g_shift_limit=[0, 0],
                                      b_shift_limit=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image

def cyan_shift(input_image_numpy, parameter=50):
    test_name = "cyan_shift"
    description = ''
    parameter = int(round(parameter))
    transform = A.Compose([A.RGBShift(r_shift_limit=[0, 0], g_shift_limit=[parameter, parameter],
                                      b_shift_limit=[parameter, parameter], always_apply=True)])
    transformed = transform(image=input_image_numpy)
    transformed_image = transformed["image"]
    return transformed_image


def generate_random_lines(imshape, slant, drop_length, rain_type):
    drops = []
    no_of_drops = 60

    if rain_type.lower() == 'drizzle':
        no_of_drops = 60
    elif rain_type.lower() == 'heavy':
        no_of_drops = 120
    elif rain_type.lower() == 'torrential':
        no_of_drops = 180

    for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        if imshape[0] - drop_length <= 0:
            y = 1
        else:
            y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops


def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, min(rain_drop[1] + drop_length,
                                                                                   image.shape[0])), drop_color, drop_width)
    image_HLS = cv2.cvtColor(image_t, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS[:, :, 1] = image_HLS[:, :, 1]
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def add_rain(image, slant=-1, drop_length_percentage=20, drop_color=(200, 200, 200),
             rain_type='None'):  ## (200,200,200) a shade of gray
    height, width = image.shape[:2]
    drop_width = max(min(width // 640, 3), 1)
    slant_extreme = slant
    drop_length = int(drop_length_percentage * height/100)
    rain_drops = generate_random_lines(image.shape, slant, drop_length, rain_type)
    output = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
    return output



def rain_drizzle_slant_0(input_image_numpy, parameter=50):
    test_name = "rain_drizzle_slant_0"
    description = 'Drizzle rain simulation. The parameter controls the length of the raindrops proportional to image ' \
                  'height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="drizzle", drop_length_percentage=parameter, slant=0)
    return transformed_image


def rain_drizzle_slant_10(input_image_numpy, parameter=50):
    test_name = "rain_drizzle_slant_10"
    description = 'Drizzle rain simulation. The parameter controls the length of the rain drops proportional to image ' \
                  'height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="drizzle", drop_length_percentage=parameter, slant=10)
    return transformed_image


def rain_drizzle_slant_neg_10(input_image_numpy, parameter=50):
    test_name = "rain_drizzle_slant_neg_10"
    description = 'Drizzle rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="drizzle", drop_length_percentage=parameter, slant=-10)
    return transformed_image


def rain_drizzle_slant_20(input_image_numpy, parameter=50):
    test_name = "rain_drizzle_slant_20"
    description = 'Drizzle rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="drizzle", drop_length_percentage=parameter, slant=20)
    return transformed_image


def rain_drizzle_slant_neg_20(input_image_numpy, parameter=50):
    test_name = "rain_drizzle_slant_neg_20"
    description = 'Drizzle rain simulation. The parameter controls the length of the rain drops proportional to image ' \
                  'height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="drizzle", drop_length_percentage=parameter, slant=-20)
    return transformed_image


def rain_heavy_slant_0(input_image_numpy, parameter=50):
    test_name = "rain_heavy_slant_0"
    description = 'Heavy rain simulation. The parameter controls the length of the rain drops proportional to image ' \
                  'height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="heavy", drop_length_percentage=parameter, slant=0)
    return transformed_image


def rain_heavy_slant_10(input_image_numpy, parameter=50):
    test_name = "rain_heavy_slant_10"
    description = 'Heavy rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="heavy", drop_length_percentage=parameter, slant=10)
    return transformed_image


def rain_heavy_slant_neg_10(input_image_numpy, parameter=50):
    test_name = "rain_heavy_slant_neg_10"
    description = 'Heavy rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="heavy", drop_length_percentage=parameter, slant=-10)
    return transformed_image


def rain_heavy_slant_20(input_image_numpy, parameter=50):
    test_name = "rain_heavy_slant_20"
    description = 'Heavy rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="heavy", drop_length_percentage=parameter, slant=20)
    return transformed_image


def rain_heavy_slant_neg_20(input_image_numpy, parameter=50):
    test_name = "rain_heavy_slant_neg_20"
    description = 'Heavy rain simulation. The parameter controls the length of the rain drops proportional to image ' \
                  'height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="heavy", drop_length_percentage=parameter, slant=-20)
    return transformed_image


def rain_torrential_slant_0(input_image_numpy, parameter=50):
    test_name = "rain_torrential_slant_0"
    description = 'Torrential rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="torrential", drop_length_percentage=parameter, slant=0)
    return transformed_image


def rain_torrential_slant_10(input_image_numpy, parameter=50):
    test_name = "rain_torrential_slant_10"
    description = 'Torrential rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="torrential", drop_length_percentage=parameter, slant=10)
    return transformed_image


def rain_torrential_slant_neg_10(input_image_numpy, parameter=50):
    test_name = "rain_torrential_slant_neg_10"
    description = 'Torrential rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="torrential", drop_length_percentage=parameter, slant=-10)
    return transformed_image


def rain_torrential_slant_20(input_image_numpy, parameter=50):
    test_name = "rain_torrential_slant_20"
    description = 'Torrential rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="torrential", drop_length_percentage=parameter, slant=20)
    return transformed_image


def rain_torrential_slant_neg_20(input_image_numpy, parameter=50):
    test_name = "rain_torrential_slant_neg_20"
    description = 'Torrential rain simulation. The parameter controls the length of the rain drops proportional to image' \
                  ' height. For example, a parameter of 3 will produce raindrops of length equal to 3% of the image height.'
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1
    if parameter > 100:
        parameter = 100
    transformed_image = add_rain(input_image_numpy, rain_type="torrential", drop_length_percentage=parameter, slant=-20)
    return transformed_image



def GridDistortion_RepeatEdge(
    img,
    num_steps=5,
    distort_limit=1,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_WRAP,
    value=None,
):
    """
    Perform grid distortion on an input image. Repeat the last pixel of the image on each axis if the size of the
     distorted image is smaller than the original image size.
    """
    xsteps = [1 + uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]
    x_step = width // num_steps
    xx = np.zeros(width, np.float32)

    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = [min(max(0, item), width - 1) for item in np.linspace(prev, cur, end - start)]
        prev = cur

    for i, _ in enumerate(xx[:-1]):
        if xx[i + 1] < xx[i]:
            xx[i + 1:] = xx[i]
        if _ > width:
            xx[i:] = width - 1
    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = [min(max(0, item), height - 1) for item in np.linspace(prev, cur, end - start)]
        prev = cur

    for i, _ in enumerate(yy[:-1]):
        if yy[i + 1] < yy[i]:
            yy[i + 1:] = yy[i]
        if _ > height:
            yy[i:] = height - 1
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


def GridDistortion_Resize(
    img,
    num_steps=5,
    distort_limit=1,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_WRAP,
    value=None,
):
    """
    Perform grid distortion on an input image. Resize the distorted image to the original size if it is smaller than the
     original image.
    """
    xsteps = [1 + uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]
    x_step = width // num_steps
    xx = np.zeros(width, np.float32)

    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = [max(0, item) for item in np.linspace(prev, cur, end - start)]
        prev = cur

    for i, _ in enumerate(xx[:-1]):
        if xx[i + 1] < xx[i]:
            xx[i + 1:] = 0
        if _ > width:
            xx[i:] = 0

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = [max(0, item) for item in np.linspace(prev, cur, end - start)]
        prev = cur

    for i, _ in enumerate(yy[:-1]):
        if yy[i + 1] < yy[i]:
            yy[i + 1:] = 0
        if _ > height:
            yy[i:] = 0

    xx = np.array([x for x in xx if x != 0], dtype="float32")
    yy = np.array([y for y in yy if y != 0], dtype="float32")
    if len(xx) == 0:
        xx = np.array(range(width))
    if len(yy) == 0:
        yy = np.array(range(height))
    xx = cv2.resize(xx, dsize=(1, width))
    yy = cv2.resize(yy, dsize=(1, height))
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


def grid_distortion_repeat_edge(input_image_numpy, parameter=1):
    test_name = "grid_distortion_repeat_edge"
    description = "Perform grid distortion on an input image. Repeat the last pixel of the image on each axis if the" \
                  " size of the distorted image is smaller than the original image size."
    if parameter < 0:
        parameter = 0

    transformed_image = GridDistortion_RepeatEdge(input_image_numpy, distort_limit=parameter, border_mode=cv2.BORDER_WRAP)
    return transformed_image

def grid_distortion_resize(input_image_numpy, parameter=1):
    test_name = "grid_distortion_resize"
    description = "Perform grid distortion on an input image. Resize the distorted image to the original size if it is " \
                  "smaller than the original image."
    if parameter < 0:
        parameter = 0

    transformed_image = GridDistortion_Resize(input_image_numpy, distort_limit=parameter, border_mode=cv2.BORDER_WRAP)
    return transformed_image

def gaussian_noise_2(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='gaussian_noise', severity=int(round(parameter)))
    test_name = "gaussian_noise"
    description = ''
    return transformed_image

def shot_noise(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='shot_noise', severity=int(round(parameter)))
    test_name = "shot_noise"
    description = ''
    return transformed_image

def impulse_noise(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='impulse_noise', severity=int(round(parameter)))
    test_name = "shot_noise"
    description = ''
    return transformed_image

def defocus_blur(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='defocus_blur', severity=int(round(parameter)))
    test_name = "defocus_blur"
    description = ''
    return transformed_image

def motion_blur_2(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='motion_blur', severity=int(round(parameter)))
    test_name = "motion_blur"
    description = ''
    return transformed_image

def snow(input_image_numpy, parameter=1):
    transformed_image = corrupt(input_image_numpy, corruption_name='snow', severity=int(round(parameter)))
    test_name = "snow"
    description = 'Snow description'
    return transformed_image

def frost(input_image_numpy, parameter=2):
    global small_count
    if input_image_numpy.shape[0] < 32 or input_image_numpy.shape[1] < 32:
        print("image is smaller than 32x32")
        small_count += 1
        cv2.imwrite(f'small_{small_count}.jpg', input_image_numpy)
        return input_image_numpy
    transformed_image = corrupt(input_image_numpy, corruption_name='frost', severity=int(round(parameter)))
    test_name = "frost"
    description = 'Frost description'
    return transformed_image

def fog(input_image_numpy, parameter=1):
    transformed_image = corrupt(input_image_numpy, corruption_name='fog', severity=int(round(parameter)))
    test_name = "fog"
    description = ''
    return transformed_image

def brightness(input_image_numpy, parameter=3):
    transformed_image = corrupt(input_image_numpy, corruption_name='brightness', severity=int(round(parameter)))
    test_name = "brightness"
    description = ''
    return transformed_image

def contrast(input_image_numpy, parameter=1):
    transformed_image = corrupt(input_image_numpy, corruption_name='contrast', severity=int(round(parameter)))
    test_name = "contrast"
    description = ''
    return transformed_image

def elastic_transform(input_image_numpy, parameter=5):
    transformed_image = corrupt(input_image_numpy, corruption_name='elastic_transform', severity=int(round(parameter)))
    test_name = "elastic_transform"
    description = ''
    return transformed_image

def pixelate(input_image_numpy, parameter=5):
    transformed_image = corrupt(input_image_numpy, corruption_name='pixelate', severity=int(round(parameter)))
    test_name = "pixelate"
    description = ''
    return transformed_image

def jpeg_compression(input_image_numpy, parameter=5):
    transformed_image = corrupt(input_image_numpy, corruption_name='jpeg_compression', severity=int(round(parameter)))
    test_name = "jpeg_compression"
    description = ''
    return transformed_image


def ConvexHullBlack(input_image_array, parameter):
    # applying some pre-processing to the image (gray scaling and blurring)
    gray_scaled = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2GRAY)
    pre_processed = cv2.blur(gray_scaled, (3, 3))
    # detecting edges using Canny edge detection algorithm according to the input parameter
    canny_edges = cv2.Canny(pre_processed, parameter, parameter * 1.5)
    # finding the contours
    edge_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filling in the convex hull list
    convex_hulls = []
    for i in range(len(edge_contours)):
        hull = cv2.convexHull(edge_contours[i])
        convex_hulls.append(hull)
    mask = np.zeros(input_image_array.shape, dtype='uint8')
    # mask convex areas
    mask = cv2.drawContours(mask, convex_hulls, -1, (255, 255, 255), thickness=cv2.FILLED)
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # apply mask on input image
    transformed_image = cv2.bitwise_and(input_image_array, input_image_array, mask=mask)
    return transformed_image


def ConvexHullWhite(input_image_array, parameter):
    # applying some pre-processing to the image (gray scaling and blurring)
    gray_scaled = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2GRAY)
    pre_processed = cv2.blur(gray_scaled, (3, 3))
    # detecting edges using Canny edge detection algorithm according to the input parameter
    canny_edges = cv2.Canny(pre_processed, parameter, parameter * 1.5)
    # finding the contours
    edge_contours, _ = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filling in the convex hull list
    convex_hulls = []
    for i in range(len(edge_contours)):
        hull = cv2.convexHull(edge_contours[i])
        convex_hulls.append(hull)
    mask = np.zeros(input_image_array.shape, dtype='uint8')
    # mask convex areas
    mask = cv2.drawContours(mask, convex_hulls, -1, (255, 255, 255), thickness=cv2.FILLED)
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # apply mask on input image
    result = cv2.bitwise_and(cv2.bitwise_not(input_image_array), cv2.bitwise_not(input_image_array), mask=mask)
    # invert the background color to white
    transformed_image = cv2.bitwise_not(result)
    return transformed_image


def convex_hull_black(input_image_array, parameter=30):
    test_name = "convex_hull_black"
    description = "Keep the convex areas of an image based on input threshold and color the rest of the image black." \
                  "Range: [1, INF). Default: [20, 100]"
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1

    transformed_image = ConvexHullBlack(input_image_array, parameter)
    return transformed_image


def convex_hull_white(input_image_array, parameter=30):
    test_name = "convex_hull_white"
    description = "Keep the convex areas of an image based on input threshold and color the rest of the image white." \
                  "Range: [1, INF). Default: [20, 100]"
    parameter = int(parameter)
    if parameter < 1:
        parameter = 1

    transformed_image = ConvexHullWhite(input_image_array, parameter)
    return transformed_image


def GammaBright(input_image_array, parameter):
    # convert img to gray
    gray = cv2.cvtColor(input_image_array, cv2.COLOR_RGB2GRAY)

    mean = np.mean(gray)
    starting_point = mean / 255

    alpha = starting_point + (parameter / 25)
    gamma = math.log(alpha * 255) / math.log(mean)

    # do gamma correction
    transformed_image = np.power(input_image_array, gamma).clip(0, 255).astype(np.uint8)
    return transformed_image


def GammaDark(input_image_array, parameter):
    # convert img to gray
    gray = cv2.cvtColor(input_image_array, cv2.COLOR_RGB2GRAY)

    mean = np.mean(gray)
    starting_point = mean / 255

    alpha = starting_point - (parameter / 25)
    gamma = math.log(alpha * 255) / math.log(mean)

    # do gamma correction
    transformed_image = np.power(input_image_array, gamma).clip(0, 255).astype(np.uint8)
    return transformed_image


def gamma_dark(input_image_numpy, parameter=3.5):
    test_name = "gamma_dark"
    description = ""
    if parameter > 5:
        parameter = 5
    elif parameter < 0:
        parameter = 0
    transformed_image = GammaDark(input_image_numpy, parameter)
    return transformed_image


def gamma_bright(input_image_numpy, parameter=6):
    test_name = "gamma_bright"
    description = ""
    if parameter < 0:
        parameter = 0
    transformed_image = GammaBright(input_image_numpy, parameter)
    return transformed_image


def edge_function(input_image_numpy, parameter):
    transformed_image = np.array(input_image_numpy, dtype=np.uint8)
    if parameter != 0:
        edges = cv2.Canny(input_image_numpy, 100, 150)
        # sdjust the parameter to have a good range
        parameter_adj = int(102 - 10 * parameter)
        kernel = np.ones((parameter_adj, parameter_adj), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        stacked_edges = np.stack((edges,) * 3, axis=-1)
        transformed_image = np.full(input_image_numpy.shape, 0, dtype=np.uint8)
        indices_edges = stacked_edges == 255
        transformed_image[indices_edges] = input_image_numpy[indices_edges]
    return transformed_image


def inverted_edge_function(input_image_numpy, parameter):
    transformed_image = np.array(input_image_numpy, dtype=np.uint8)
    if parameter != 0:
        edges = cv2.Canny(input_image_numpy, 100, 150)
        kernel = np.ones((int(parameter), int(parameter)), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        stacked_edges = np.stack((edges,) * 3, axis=-1)
        transformed_image = np.full(input_image_numpy.shape, 0, dtype=np.uint8)
        indices_edges = stacked_edges == 0
        transformed_image[indices_edges] = input_image_numpy[indices_edges]
    return transformed_image


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Code inspired from https://github.com/Gil-Mor/iFish
    Get normalized x, y pixel coordinates from the original image and return normalized
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion * (radius ** 2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion * (radius ** 2))), source_y / (1 - (distortion * (radius ** 2)))


def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):
            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2 * x - w) / w), float((2 * y - h) / h)

            # get xn and yn distance from normalized center
            rd = math.sqrt(xnd ** 2 + ynd ** 2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)


def postprocess(img):
    # remove empty space around images and resize them to the original size
    min_x = min(np.where(img[:, :, 3] == 255)[0])
    min_y = min(np.where(img[:, :, 3] == 255)[1])
    max_x = max(np.where(img[:, :, 3] == 255)[0])
    max_y = max(np.where(img[:, :, 3] == 255)[1])
    img_postprocess = cv2.resize(img[min_x:max_x, min_y:max_y, :], [img.shape[1], img.shape[0]])

    return np.delete(img_postprocess, obj=3, axis=2)


def edge(input_image_numpy, parameter=5):
    test_name = "edge"
    description = ''
    transformed_image = edge_function(input_image_numpy, parameter)
    return transformed_image


def inverted_edge(input_image_numpy, parameter=50):
    test_name = "inverted_edge"
    description = ''
    transformed_image = inverted_edge_function(input_image_numpy, parameter)
    return transformed_image


def fisheye(input_image_numpy, parameter=0.5):
    test_name = "fisheye"
    description = ''
    transformed_image = postprocess(fish(input_image_numpy, parameter))
    return transformed_image


def pincushion(input_image_numpy, parameter=0.5):
    test_name = "pincushion"
    description = ''
    transformed_image = postprocess(fish(input_image_numpy, -parameter))
    return transformed_image


def camera_shake_upper_left(input_image_numpy, parameter=5):
    test_name = "camera_shake_upper_left"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(0, input_image_numpy.shape[0] - parameter):
        for col in range(0, input_image_numpy.shape[1] - parameter):
            temp[row, col] = input_image_numpy[row + parameter, col + parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_up(input_image_numpy, parameter=5):
    test_name = "camera_shake_up"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(0, input_image_numpy.shape[0] - parameter):
        for col in range(0, input_image_numpy.shape[1] - parameter):
            temp[row, col] = input_image_numpy[row + parameter, col]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_upper_right(input_image_numpy, parameter=5):
    test_name = "camera_shake_upper_right"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(0, input_image_numpy.shape[0] - parameter):
        for col in range(parameter, input_image_numpy.shape[1]):
            temp[row, col] = input_image_numpy[row + parameter, col - parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_right(input_image_numpy, parameter=5):
    test_name = "camera_shake_right"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(parameter, input_image_numpy.shape[0]):
        for col in range(parameter, input_image_numpy.shape[1]):
            temp[row, col] = input_image_numpy[row, col - parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_lower_right(input_image_numpy, parameter=5):
    test_name = "camera_shake_lower_right"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(parameter, input_image_numpy.shape[0]):
        for col in range(parameter, input_image_numpy.shape[1]):
            temp[row, col] = input_image_numpy[row - parameter, col - parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_down(input_image_numpy, parameter=5):
    test_name = "camera_shake_down"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(parameter, input_image_numpy.shape[0]):
        for col in range(parameter, input_image_numpy.shape[1]):
            temp[row, col] = input_image_numpy[row - parameter, col]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_lower_left(input_image_numpy, parameter=5):
    test_name = "camera_shake_lower_left"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(parameter, input_image_numpy.shape[0]):
        for col in range(0, input_image_numpy.shape[1] - parameter):
            temp[row, col] = input_image_numpy[row - parameter, col + parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_left(input_image_numpy, parameter=5):
    test_name = "camera_shake_left"
    description = ""
    license = 'Zetane'
    parameter = int(min(input_image_numpy.shape[0], input_image_numpy.shape[1]) * parameter / 100)
    temp = np.array(input_image_numpy, dtype=np.uint8)
    for row in range(0, input_image_numpy.shape[0] - parameter):
        for col in range(0, input_image_numpy.shape[1] - parameter):
            temp[row, col] = input_image_numpy[row, col + parameter]
    transformed_image = cv2.addWeighted(input_image_numpy, 0.7, temp, 0.3, 0.0)
    return transformed_image


def camera_shake_all_directions(input_image_numpy, parameter=1):
    test_name = "camera_shake_all_directions"
    description = ""
    license = 'Zetane'
    transformed_image = input_image_numpy.copy()
    shake = camera_shake_up(transformed_image, parameter)
    shake = camera_shake_right(shake, parameter)
    shake = camera_shake_left(shake, parameter)
    shake = camera_shake_down(shake, parameter)
    transformed_image = shake
    return transformed_image
