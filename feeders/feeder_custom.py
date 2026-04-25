import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from feeders.bone_pairs import ntu_pairs, coco17_pairs


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        p_interval=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=-1,
        normalization=False,
        debug=False,
        bone=False,
        vel=False,
        num_point=17,
        num_person=1,
    ):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.num_point = num_point
        self.num_person = num_person

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path)

        if self.split == 'train':
            x_key, y_key = 'x_train', 'y_train'
            prefix = 'train'
        elif self.split == 'test':
            x_key, y_key = 'x_test', 'y_test'
            prefix = 'test'
        else:
            raise NotImplementedError('data split only supports train/test')

        if x_key not in npz_data or y_key not in npz_data:
            raise KeyError('Expected keys x_train/x_test and y_train/y_test in npz file')

        data = npz_data[x_key]
        label = npz_data[y_key]

        # Preferred format: (N, C, T, V, M)
        if data.ndim == 5:
            self.data = data.astype(np.float32)
            self.num_point = int(self.data.shape[3])
            self.num_person = int(self.data.shape[4])
        # Alternate format: (N, T, V, C)
        elif data.ndim == 4 and data.shape[-1] in (2, 3):
            self.data = data.transpose(0, 3, 1, 2)[:, :, :, :, None].astype(np.float32)
            self.num_point = int(self.data.shape[3])
            self.num_person = 1
        # Legacy flattened format: (N, T, V*C*M)
        elif data.ndim == 3:
            n, t, vc = data.shape
            c = 3
            if vc % (c * self.num_person) != 0:
                raise ValueError('Cannot infer num_point from flattened input shape')
            v = vc // (c * self.num_person)
            self.num_point = v
            self.data = data.reshape(n, t, c, v, self.num_person).transpose(0, 2, 1, 3, 4).astype(np.float32)
        else:
            raise ValueError('Unsupported data shape. Expected 5D/4D/3D input in NPZ.')

        if label.ndim == 1:
            self.label = label.astype(np.int64)
        elif label.ndim == 2:
            self.label = np.argmax(label, axis=1).astype(np.int64)
        else:
            raise ValueError('Unsupported label shape. Expected 1D index or 2D one-hot/soft labels.')

        self.sample_name = [f'{prefix}_{i}' for i in range(len(self.data))]

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def get_mean_map(self):
        data = self.data
        n, c, t, v, m = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((n * t * m, c * v)).std(axis=0).reshape((c, 1, v, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = int(self.label[index])

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            if self.num_point == 25:
                pairs = ntu_pairs
            elif self.num_point == 17:
                pairs = coco17_pairs
            else:
                raise ValueError(f'Bone mode is not configured for num_point={self.num_point}')

            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
