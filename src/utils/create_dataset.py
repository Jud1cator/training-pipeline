from dataset_reformat import DatasetReformat, NewDatasetReformat


CLASS_MAPPING = {
    'c2': 'c1',
    'c3': 'c1',
    'c4': 'c2',
    'c5': 'c2',
    'c6': 'c2',
    'c7': 'c2',
    'c9': 'c3',
    'c10': 'c4',
    'c11': 'c5',
    'c12': 'c5',
    'c13': 'c5',
    'c8': 'c6',
}

root = '/home/judicator/ev/dvc/sbc_core/Datasets/NewClassesDataset'
dest = '/home/judicator/ev/data/vehicle_classification_6'

NewDatasetReformat(root, time_type='day', class_mapping=CLASS_MAPPING)(dest)
NewDatasetReformat(root, time_type='evening', class_mapping=CLASS_MAPPING)(dest)
NewDatasetReformat(root, time_type='night', class_mapping=CLASS_MAPPING)(dest)

additional_root = '/home/judicator/ev/dvc/sbc_core/Datasets/ObjectDetection_c1c6_person/dataset_generated'
additional_class_names = ['c2', 'c6']
DatasetReformat(additional_root, additional_class_names)(dest)
