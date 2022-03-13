import tensorflow as tf
import argparse
from tensorflow import keras
from upscaler import train_utils


def get_args():
    parser = argparse.ArgumentParser(description="Process test images using pre-trained model")
    parser.add_argument('--model', help='path to pre-trained model')
    parser.add_argument('--images', help='path to source test images')
    parser.add_argument('--output', help="Output path to export test results")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model = keras.models.load_model(args.model)
    train_utils.generate_test_images(args.images, args.output, model)
