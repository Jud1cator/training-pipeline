import os
import cv2
import json


class DatasetReformat:
    """
        Creates new directory for a c1-c6 blob dataset.
        Crops and copies the blobs from image to corresponding class folder.
    """

    def __init__(self, root_path, class_names):
        self.root = root_path
        self.class_names = class_names

    def __call__(self, destination, *args, **kwargs):
        count_dict = {cn: 0 for cn in self.class_names}
        for file in os.listdir(self.root):
            if file[-4:] != 'json':
                continue
            with open(f'{self.root}/{file}') as fp:
                objects = json.load(fp)
            img_file = f'{self.root}/{file[:-4]}jpg'
            img = cv2.imread(img_file)
            for detection in objects:
                label = detection['class']
                if label not in self.class_names:
                    continue
                polygon = detection['polygon']
                if len(polygon) < 3:
                    continue
                xs, ys = [int(coordinate[0]) for coordinate in polygon], \
                         [int(coordinate[1]) for coordinate in polygon]
                box = ([min(xs), min(ys), max(xs), max(ys)])
                if count_dict[label] == 0:
                    os.makedirs(f'{destination}/{label}', exist_ok=True)
                crop = img[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(f'{destination}/{label}/{file[:-4]}_{count_dict[label]}.jpg', crop)
                count_dict[label] += 1
        print(f'Class distribution: {count_dict}')


class NewDatasetReformat:
    """
           Creates new directory for a c1-c6 blob dataset.
           Crops and copies the blobs from image to corresponding class folder according to gt.
    """

    def __init__(self, root_path, time_type, class_mapping):
        self.root = root_path
        self.time_type = time_type
        self.class_mapping = class_mapping

    def __call__(self, destination):
        count_dict = {cn: 0 for cn in self.class_mapping.values()}
        for file in os.listdir(f'{self.root}/gt/{self.time_type}'):
            if file[-4:] != 'json':
                continue
            with open(f'{self.root}/gt/{self.time_type}/{file}') as fp:
                objects = json.load(fp)
            img_file = f'{self.root}/images/{file[:-4]}jpg'
            img = cv2.imread(img_file)
            for detection in objects:
                label = detection['class']
                if label not in self.class_mapping:
                    continue
                label = self.class_mapping[label]
                polygon = detection['polygon']
                if len(polygon) < 3:
                    continue
                xs, ys = [int(coordinate[0]) for coordinate in polygon], \
                         [int(coordinate[1]) for coordinate in polygon]
                box = ([min(xs), min(ys), max(xs), max(ys)])
                if count_dict[label] == 0:
                    os.makedirs(f'{destination}/{label}', exist_ok=True)
                crop = img[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(f'{destination}/{label}/{file[:-4]}_{count_dict[label]}.jpg', crop)
                count_dict[label] += 1
        print(f'Class distribution: {count_dict}')
