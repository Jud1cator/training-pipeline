import random
import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt


# random.seed(43)

data_dir = "/home/judicator/ev/data/vehicle_classification_6_balanced/"
classes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']

# transform = A.RandomBrightnessContrast(
#     brightness_limit=(-0.5, 0.0),
#     contrast_limit=(-0.2, 0.0),
#     p=1.0
# )
transform = A.Blur(blur_limit=(6, 6), p=1.0)

fname = random.choice(os.listdir(data_dir + 'c4'))
img = cv2.imread(data_dir + 'c4/' + fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transformed = transform(image=img)['image']
plt.imshow(img)
plt.show()
plt.imshow(transformed)
plt.show()
