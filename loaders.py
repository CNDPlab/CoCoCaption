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


class HDFSet(Dataset):
    def __init__(self, set):
        file = h5py.File('processed.hdf', 'r')
        self.features = file[set]['feature']
        self.labels = file[set]['label']
        self.lenths = file[set]['lenths']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.labels[item], self.lenths[item]


def get_loaders(set, batch_size):
    dataset = HDFSet(set)
    dataloader = DataLoader(dataset, batch_size, True, drop_last=True)
    return dataloader


if __name__ == '__main__':
    pass
    # count = []
    # from tqdm import tqdm
    # loader = get_loader('val', 2)
    # for i in tqdm(loader):
    #     pass
