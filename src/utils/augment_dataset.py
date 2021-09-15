import random
import albumentations as A
import cv2
import os


random.seed(42)

data_dir = "/home/judicator/ev/data/vehicle_classification_6_balanced/"
classes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']

flip = A.HorizontalFlip(p=1.0)
flip_id = 'flip'
brightness_contrast = A.RandomBrightnessContrast(
    brightness_limit=(-0.5, 0.0),
    contrast_limit=(-0.2, 0.0),
    p=1.0
)
brightness_contrast_id = 'bc'
blur = A.Blur(blur_limit=(3, 6), p=1.0)
blur_id = 'blur'

transforms = {
    flip_id: flip,
    brightness_contrast_id: brightness_contrast,
    blur_id: blur
}

for c in classes:
    imgs = os.listdir(data_dir + c)
    if len(imgs) == 1000:
        continue
    to_add = min(1000 - len(imgs), len(imgs))
    imgs_to_transform = random.sample(imgs, to_add)
    for i in range(len(imgs_to_transform)):
        img = cv2.imread(data_dir + c + '/' + imgs_to_transform[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed_img = transforms[flip_id](image=img)['image']
        fname = imgs_to_transform[i][:-4] + '_' + flip_id + str(i) + '.jpg'
        cv2.imwrite(data_dir + c + '/' + fname, transformed_img)
    print(c + ': ' + str(len(os.listdir(data_dir + c))))
print()

for c in classes:
    imgs = os.listdir(data_dir + c)
    if len(imgs) == 1000:
        continue
    to_add = min(1000 - len(imgs), len(imgs))
    imgs_to_transform = random.sample(imgs, to_add)
    for i in range(len(imgs_to_transform)):
        img = cv2.imread(data_dir + c + '/' + imgs_to_transform[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed_img = transforms[brightness_contrast_id](image=img)['image']
        fname = imgs_to_transform[i][:-4] + '_' + brightness_contrast_id + str(i) + '.jpg'
        cv2.imwrite(data_dir + c + '/' + fname, transformed_img)
    print(c + ': ' + str(len(os.listdir(data_dir + c))))
print()

for c in classes:
    imgs = os.listdir(data_dir + c)
    if len(imgs) == 1000:
        continue
    to_add = min(1000 - len(imgs), len(imgs))
    imgs_to_transform = random.sample(imgs, to_add)
    for i in range(len(imgs_to_transform)):
        img = cv2.imread(data_dir + c + '/' + imgs_to_transform[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed_img = transforms[blur_id](image=img)['image']
        fname = imgs_to_transform[i][:-4] + '_' + blur_id + str(i) + '.jpg'
        cv2.imwrite(data_dir + c + '/' + fname, transformed_img)
    print(c + ': ' + str(len(os.listdir(data_dir + c))))
