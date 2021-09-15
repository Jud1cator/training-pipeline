import os

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image


def main(params):
    classes = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6')
    removed = 0
    for c in classes:
        print(f"Scanning class {c}...")
        path = Path(params.data_path) / c
        for img_name in os.listdir(path):
            img_path = path / img_name
            img = Image.open(img_path)
            if img.size[0] / img.size[1] > 2.5 or img.size[1] / img.size[0] \
                    > 2.5:
                # print(img_path)
                removed += 1
                os.remove(img_path)
    print("Total images removed: ", removed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-path')
    params = parser.parse_args()
    main(params)
