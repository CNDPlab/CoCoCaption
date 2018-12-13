from h5py import File
import numpy as np
from torch.utils.data import TensorDataset, ConcatDataset
from concurrent.futures import ProcessPoolExecutor




feature = np.ones((150,300))

np.split(feature, )



with File('test.hdf', 'w') as file:
    file.create_dataset('feature',)


