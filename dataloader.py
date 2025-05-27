from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

def load_data(dataset):
    if dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 70000
    elif dataset == "BBCSport":
        dataset=BBCSport('./data/')
        dims = [3183,3203]
        view = 2
        class_num = 5
        data_size = 544
    elif dataset == "YouTubeFace":
        dataset=YouTubeFace('./data/')
        dims = [64,512,64,647,838]
        view = 5
        class_num = 31
        data_size = 101499
    elif dataset == "cifar100":
        dataset=cifar100('./data/')
        dims = [512,2048,1024]
        view = 3
        class_num = 100
        data_size = 50000
    elif dataset == "cifar10":
        dataset=cifar10('./data/')
        dims = [512,2048,1024]
        view = 3
        class_num = 10
        data_size = 50000
    elif dataset == "STL10":
        dataset=STL10('./data/')
        dims = [1024,512,2048]
        view = 3
        class_num = 10
        data_size = 13000
    elif dataset == "Caltech101":
        dataset=Caltech101('./data/')
        dims = [48,40,254,1984,512,928]
        view = 6
        class_num = 102
        data_size = 9144
    elif dataset == "NUSWIDEOBJ":
        dataset=NUSWIDEOBJ('./data/')
        dims = [65,226,145,74,129]
        view = 5
        class_num = 31
        data_size = 30000
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num



class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech101():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101.mat')
        self.Y = data['Y'].T.astype(np.int32).reshape(9144,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)
    def __len__(self):
        return 9144
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
       
class STL10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'STL10.mat')
        self.Y = data['Y'].T.astype(np.int32).reshape(13000,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 13000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUSWIDEOBJ():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NUSWIDEOBJ.mat')
        self.Y = data['Y'].T.astype(np.int32).reshape(30000,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 30000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class YouTubeFace():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YouTubeFace_sel_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(101499,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 101499
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class cifar100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
        
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class cifar10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
        
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path + 'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path + 'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path + 'MFCC.npy').astype(np.float32)
        self.labels = np.load(path + 'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
            x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class BBCSport():
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BBCSport.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BBCSport.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BBCSport.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels
    def __len__(self):
        return 544
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NoisyMNIST(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NoisyMNIST.mat')
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        X_list = []
        Y_list = []
        X_list.append(np.concatenate([train.images1, tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([train.images2, tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate(
            [np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        print(Y_list[0])
        x1 = X_list[0]
        x2 = X_list[1]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(Y_list[0])
        self.images1 = xx1.astype(np.float32)
        self.images2 = xx2.astype(np.float32)
        self.labels = Y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x1 = self.images1[idx].reshape(784)
        x2 = self.images2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.labels[idx], torch.from_numpy(np.array(idx)).long()

class DataSet_NoisyMNIST(object):
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()




