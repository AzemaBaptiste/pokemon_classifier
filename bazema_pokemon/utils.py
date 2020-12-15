"""
Argument parser
"""
import argparse

import cv2
from torchvision import transforms


def create_parser():
    """
    Parser
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        help='File to process',
        required=True
    )
    return parser


def parse_args(args):
    """
    Parse arguments
    :param args: raw args
    :return: Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args(args=args)


def torch_transformations():
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transform = {'train': transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)

    ]), 'val': transforms.Compose([
        transforms.Resize(size=64),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)

    ]), 'test': transforms.Compose([
        transforms.Resize(size=64),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ]),
    }
    return transform, std_nums, mean_nums


def face_detector(img_path):
    """returns "True" if face is detected in image stored at img_path"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path = 'resources/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
