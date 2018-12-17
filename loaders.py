import torchvision.datasets as dset
from Predictor.Utils import picture_tranform_func
import h5py
from torch.utils.data import TensorDataset, Dataset, DataLoader


def get_dataset(data_set):
    assert data_set in ['train', 'val', 'test']
    data_root = f'cocoapi/images/{data_set}2017/'
    annfile = f'cocoapi/annotations/captions_{data_set}2017.json'
    dataset = dset.CocoCaptions(
        root=data_root,
        annFile=annfile,
        transform=picture_tranform_func
    )
    return dataset


class HDFSet(TensorDataset):
    def __init__(self, set):
        super(HDFSet, self).__init__()
        self.set = set
        with h5py.File('processed.hdf', 'r') as reader:
            le = len(reader[self.set]['feature'])
        self.lenth = le

    def __len__(self):
        #return 10
        return self.lenth

    def __getitem__(self, item):
        with h5py.File('processed.hdf', 'r') as reader:
            feature = reader[self.set]['feature'][item]
            label = reader[self.set]['label'][item]
            lenth = reader[self.set]['lenth'][item]
        return feature, label, lenth


def get_loaders(set, batch_size, num_works):
    dataset = HDFSet(set)
    dataloader = DataLoader(dataset, batch_size, True, drop_last=True, num_workers=num_works)
    return dataloader
def get_loaders2(set, batch_size, num_works):
    dataset = HDFSet1(set)
    dataloader = DataLoader(dataset, batch_size, True, drop_last=True, num_workers=num_works)
    return dataloader

def get_loaders3(set, batch_size, num_works):
    dataset = HDFSet2(set)
    dataloader = DataLoader(dataset, batch_size, True, drop_last=True, num_workers=num_works)
    return dataloader

class HDFSet1(TensorDataset):
    def __init__(self, set):
        super(HDFSet1, self).__init__()
        self.set = set
        with h5py.File('processed.hdf', 'r') as reader:
            le = len(reader[self.set]['feature'])
        self.lenth = le

    def __len__(self):
        #return 10
        return self.lenth

    def __getitem__(self, item):
        with h5py.File('processed.hdf', 'r') as reader:
            feature = reader[self.set]['feature'][item]
            label = reader[self.set]['label'][item]
            lenth = reader[self.set]['lenth'][item]
        return feature, label, lenth

class HDFSet2(TensorDataset):
    def __init__(self, set):
        super(HDFSet2, self).__init__()
        self.set = set
        with h5py.File('processed.hdf', 'r') as reader:
            le = len(reader[self.set]['feature'])
            self.lenth = le
            self.feature = reader[self.set]['feature'].value
            self.label = reader[self.set]['label'].value
            self.lenth = reader[self.set]['lenth'].value

    def __len__(self):
        #return 10
        return self.lenth

    def __getitem__(self, item):

        return self.feature[item], self.label[item], self.lenth[item]


if __name__ == '__main__':
    loader1 = get_loaders2('train',20,10)
    from tqdm import tqdm

    for i in tqdm(loader1):
        pass
    loader2 = get_loaders3('train',20,10)
    for i in tqdm(loader2):
        pass