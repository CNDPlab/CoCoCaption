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

    def __len__(self):
        with h5py.File('processed.hdf', 'r') as reader:
            le = len(reader[self.set]['feature'])
        return le

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


if __name__ == '__main__':
    pass
    # count = []
    # from tqdm import tqdm
    # loader = get_loader('val', 2)
    # for i in tqdm(loader):
    #     pass
#
# from tqdm import tqdm
# from loaders import get_loaders
#
# val = get_loaders('val',32,2)
# for i in tqdm(val):
#     pass