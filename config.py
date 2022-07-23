import os
import sys

def define_arguments():
    import argparse

    print("AYA")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/demo",
        help="Data to infer"
    )
    parser.add_argument(
        "--training_data_path",
        type=str,
        default="/data/ocr/",
        help="training dataset to use",
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default="/data/ocr/test/",
        help="validation dataset to use",
    )
    parser.add_argument(
        "--max_image_large_side", type=int, default=1280, help="max image size of training"
    )
    parser.add_argument(
        "--max_text_size",
        type=int,
        default=800,
        help="if the text in the input image is bigger than this, then we resize"
        "the image according to this",
    )
    parser.add_argument(
        "--min_text_size",
        type=int,
        default=10,
        help="if the text size is smaller than this, we ignore it during training",
    )
    parser.add_argument(
        "--min_crop_side_ratio",
        type=float,
        default=0.1,
        help="when doing random crop from input image, the" "min length of min(H, W",
    )
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint/ckpt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--base_model", type=str, default="mobilenet")
    parser.add_argument("--unfreeze", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--dont_write", action="store_true", default=False)
    parser.add_argument("--write_timer", action="store_true", default=False)
    return parser.parse_args()

cfg = {}

if int(os.getenv('APP_RUNNING') or 0) != 1:
    cfg = define_arguments()
