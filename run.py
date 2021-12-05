from module import cls
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type = str, help='please image -input')
    args = parser.parse_args()
    cls.predictions_classes(args.i)
