from typing import Optional, Tuple, Union, List
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CocoDetection
from torch.utils.data import random_split, Dataset
from utils.dataset import get_annotations_info
import warnings

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class DetectionDataModule(LightningDataModule):

    def __init__(
        self,
        images_dir: str,
        annotation_files: Union[List[str], str],
        input_shape: Optional[Tuple[int, int]],
        batch_size: int,
        val_split: float = 0.3,
        test_split: float = 0.0,
        num_workers: int = 8,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            :param data_dir: path to folder with data
            :param input_shape: model input image resolution
            :param batch_size: the batch size
            :param num_workers: how many workers to use for loading data
            :param test_data_dir: path to folder with test data
            :param shuffle: if true shuffles the data every epoch
            :param pin_memory: if true, the data loader will copy Tensors 
                               into CUDA pinned memory before returning them
            :param drop_last: If true drops the last incomplete batch
        """
        super(DetectionDataModule, self).__init__(*args, **kwargs)
        self.images_dir = images_dir
        self.annotations = annotation_files
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        assert val_split + test_split < 1, f"Provided ratios for val and test splits \
                                             should not be greater than 1.0, got {val_split} for val \
                                             and {test_split} for test"

        self.parse_annotations(val_split, test_split)
        
    
    def parse_annotations(self, val_ratio: float, test_ratio: float):
        """Parse provided annotations. 
            If there are several annotations files tries to parse them 
            by the type (train, val, test). 
            Otherwise, parse only one file and split it by provided 
            ratios for sets.
        """
        if type(self.annotations) is list:
            # parse filenames to get internal info
            info, parsed = get_annotations_info(self.annotations)

            if info['val']:
                warnings.warn('[!] val_split will not be considered, because validation set is already provided')

            if info['test']:
                warnings.warn('[!] test_split will not be considered, because test set is already provided')
            
            if info['unknown']:
                warnings.warn('Unknown annotations are found. This annotations will be concidered as train set')
            

        elif type(self.annotations) is str:
            # load annotation file and split it on train, val, test
            dataset = CocoDetection(self.images_dir, annFile=self.annotations)
            total_len = len(dataset)

            # compute lengthsfor sets based on ratios
            val_len = int(total_len * val_ratio)
            test_len = int(total_len * test_ratio)
            train_len = total_len - val_len - test_len

            lengths = [train_len, val_len, test_len]

            train_data, val_data, test_data = random_split(dataset, lengths)

            # generate datasets out of subsets with denoted transforms
            self.train_set = DatasetFromSubset(
                train_data, transform=self.train_transform()
            )

            self.val_set = DatasetFromSubset(
                val_data, transform=self.val_transform()
            )

            self.test_set = DatasetFromSubset(
                test_data, transform=self.val_transform()
            )
    
    def train_transform(self):
        pass

    def val_transform(self):
        pass




# if __name__ == "__main__":
    # test_datamodule = DetectionDataModule()