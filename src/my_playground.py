import os
import argparse
from tools import *
from model import *
from rectification import *
from classification import *
from detection import *
from measurement import *

from my_utils import *


def main(input):
    # TODO: Change image input format
    path_to_input_image = "{}".format(input)
    print(f"Loaded image: {path_to_input_image}")

    # TODO: Test change resize_value
    resize_value = 256
    # resize_value = 512

    # TODO: Add custom result path
    # path_to_my_results = "../my_results/PalmAll-Bright60-300-400"
    # path_to_my_results = "../my_results/PalmAll-Cross-300-400"
    # path_to_my_results = "../my_results/PalmAll-MultiLine-300-400"
    path_to_my_results = "../my_results/etc"
    check_path_compatibility(path_to_my_results)


    path_to_model = "checkpoint/checkpoint_aug_epoch70.pth"

    # TODO: Image name format
    file_name = get_filename_without_extension(path_to_input_image)
    output_format = f"{path_to_my_results}/{file_name}"
    print(f"image file name: {file_name}")
    logger.info(f"Image: {file_name}")

    path_to_my_process = f"{path_to_my_results}/process"
    check_path_compatibility(path_to_my_process)
    output_process_format = f"{path_to_my_process}/{file_name}"

    path_to_predicted_mask = f"{path_to_my_results}/predicted-mask"
    check_path_compatibility(path_to_predicted_mask)
    predicted_mask_format = f"{path_to_predicted_mask}/{file_name}"

    path_to_comparison = f"{path_to_my_results}/compare"
    check_path_compatibility(path_to_comparison)
    result_comparison_format = f"{path_to_comparison}/{file_name}-Compare"

    path_to_clean_image = f"{output_process_format}-A-removed_background.jpg"
    path_to_warped_image = f"{output_process_format}-B-warped_palm.jpg"
    path_to_warped_image_clean = f"{output_process_format}-C-warped_palm_clean.jpg"
    path_to_warped_image_mini = f"{output_process_format}-B-warped_palm_mini.jpg"
    path_to_warped_image_clean_mini = f"{output_process_format}-B-warped_palm_clean_mini.jpg"
    path_to_palmline_image = f"{predicted_mask_format}-palm_lines.png"

    path_to_result = f"{path_to_my_results}/result"
    check_path_compatibility(path_to_result)
    result_format = f"{path_to_result}/{file_name}-Result.jpg"

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)
    logger.info(f"\t remove background, \t save at {path_to_clean_image}")

    # 1. Palm image rectification
    warp_result = warp(path_to_input_image, path_to_warped_image)
    logger.info(f"\t save wrap, \t\t\t save at {path_to_warped_image}")

    if warp_result is None:
        print_error()
        # MyDebug: warp image
        print("ERROR: image can't warp")
        logger.warning(f"\t image can't warp")
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
        check_path_exists(path_to_palmline_image)

        # TODO: Custom save
        save_comparison(path_to_warped_image, path_to_palmline_image, result_comparison_format)
        logger.info(f"\t save comparison, \t\t save at {result_comparison_format}")

        # 3. Line classification
        lines = classify(path_to_palmline_image)
        print(f"len(lines): {len(lines)}")
        logger.info(f"\t number of detected line: {len(lines)}")

        # 4. Length measurement
        im, contents = measure(path_to_warped_image_mini, lines)
        if im is None:
            logger.error(f"\t image not loaded")
        else:
            logger.info(f"\t save measurement, \t\t save at {path_to_warped_image_mini}")


        # 5. Save result
        save_result(im, contents, resize_value, result_format)
        # check_path_exists(result_format)

def get_images_dataset(images_folder: str) -> list[str]:
    return sorted(os.listdir(images_folder))

def run_images_dataset(parser: argparse.ArgumentParser):
    # images_path = "/home/selfapp/Work_Sun/SunGit/Crack-Segmentation/data/Palm/PalmAll"
    # images_path = "/home/selfapp/Work_Sun/SunGit/Crack-Segmentation/output/dataset/palm-thick3-multiline/input"
    # images_path = "/home/selfapp/Work_Sun/SunGit/data-ubuntu/PalmAll-Bright60"
    # images_path = "../data/Palm/PalmAll-Bright60-300-400"
    # images_path = "../data/Palm/thick3-category/cross/input"
    # images_path = "../data/Palm/thick3-category/multiline/input"
    images_path = "../data/test-palm"

    images_file = get_images_dataset(images_path)
    logger.info(f"----- Start render from images path: {images_path} -----")
    for image_file in images_file:
        # if image_file in ['IMG_FEMALE_0016.jpg']:
        #     continue
        args = parser.parse_args(["--input", f"{images_path}/{image_file}"])
        main(args.input)

def run_image_dataset(parser: argparse.ArgumentParser, image_path: str):
    args = parser.parse_args(["--input", image_path])
    main(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="the path to the input")

    # TODO: Add input into parser
    # args = parser.parse_args()

    # image_path = "../data/Palm/original/IMG_0009.JPG"
    image_path = "../data/Alex0301.JPG"

    # TODO: Add change directory
    change_to_project_path(__file__)

    # TODO: use images or image only
    # run_images_dataset(parser)

    run_image_dataset(parser, image_path)
