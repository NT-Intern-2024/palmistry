import os
import argparse
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *

# TODO: Custom import
from my_utils import *

def main(input):
    # TODO: edit input format
    path_to_input_image = "{}".format(input)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    resize_value = 256
    path_to_clean_image = "results/palm_without_background.jpg"
    path_to_warped_image = "results/warped_palm.jpg"
    path_to_warped_image_clean = "results/warped_palm_clean.jpg"
    path_to_warped_image_mini = "results/warped_palm_mini.jpg"
    path_to_warped_image_clean_mini = "results/warped_palm_clean_mini.jpg"
    path_to_palmline_image = "results/palm_lines.png"
    path_to_model = "checkpoint/checkpoint_aug_epoch70.pth"
    path_to_result = "results/result.jpg"

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)

    # 1. Palm image rectification
    warp_result = warp(path_to_input_image, path_to_warped_image)
    if warp_result is None:
        print_error()
    else:
        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(
            path_to_warped_image,
            path_to_warped_image_clean,
            path_to_warped_image_mini,
            path_to_warped_image_clean_mini,
            resize_value,
        )

        # 2. Principal line detection
        net = UNet(n_channels=3, n_classes=1)
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Length measurement
        # im, contents = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        # save_result(im, contents, resize_value, path_to_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="the path to the input")

    # TODO: Add parser
    # args = parser.parse_args()
    # image_path = "../data/Palm/original/IMG_0009.JPG"
    change_to_project_path(__file__)
    print(f"current path: {os.getcwd()}")
    image_path = "input/hand1.jpg"
    # image_path = "/home/selfapp/Work_Sun/SunGit/Crack-Segmentation/data/Palm/PalmAll/IMG_FEMALE_0014.jpg"
    print(f"check image path: {os.path.exists(image_path)}")
    args = parser.parse_args(["--input", image_path])
    main(args.input)
