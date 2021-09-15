import os
import shutil
import random

SEED = 42
random.seed(SEED)

classes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']

source_data_dir = '/home/judicator/d/ev/data/vehicle_classification_6_new_v2'
dest_data_dir = '/home/judicator/d/ev/data/vehicle_classification_6_new_v2_train'
test_data_dir = '/home/judicator/d/ev/data/vehicle_classification_6_new_test'

test = {}
train = {}
# for c in classes:
#     imgs = os.listdir(source_data_dir + '/' + c)
#     test[c] = random.sample(imgs, 100)
#     train[c] = [i for i in imgs if i not in test[c]]
for c in classes:
    imgs = os.listdir(source_data_dir + '/' + c)
    test_imgs = os.listdir(test_data_dir + '/' + c)
    train[c] = [i for i in imgs if i not in test_imgs]

for c in classes:
    src = source_data_dir + '/' + c + '/'
    # dest_test = test_data_dir + '/' + c + '/'
    # for i in range(len(test[c])):
    #     shutil.copyfile(src + test[c][i], dest_test + test[c][i])
    # train_len = min(1000, len(train[c]))
    # dest_train = dest_data_dir + '/' + c + '/'
    # for img in random.sample(train[c], train_len):
    #     shutil.copyfile(src + img, dest_train + img)
    dest_train = dest_data_dir + '/' + c + '/'
    for i in range(len(train[c])):
        shutil.copyfile(src + train[c][i], dest_train + train[c][i])
