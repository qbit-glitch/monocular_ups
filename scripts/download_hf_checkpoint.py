"""Download a HuggingFace Mask2Former checkpoint to ./checkpoints/."""

import argparse
import os

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Download HF Mask2Former checkpoint")
    parser.add_argument(
        "--model_id",
        default="facebook/mask2former-swin-tiny-coco-panoptic",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--save_dir",
        default="./checkpoints",
        help="Directory to save the checkpoint",
    )
    args = parser.parse_args()

    save_path = os.path.join(args.save_dir, args.model_id.replace("/", "--"))
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading {args.model_id} ...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_id)
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_id)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
