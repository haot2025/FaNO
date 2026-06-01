import sys
import os
import time

import torch
import hashlib
import numpy as np
import scipy
import re

import pynvml
from pynvml import NVMLError
# == Pytorch things

def convert_to_one_hot(labels, num_classes=None):
    """
    将标签张量转换为 one-hot 编码
    
    参数:
    labels: 形状为 [b, n] 的整数张量，每个元素表示类别索引
    num_classes: 类别数量，如果不提供则使用 labels 中的最大值+1
    
    返回:
    one_hot: 形状为 [b, n, c] 的 one-hot 编码张量
    """
    # 获取输入形状
    b, n = labels.shape
    
    # 确定类别数量
    if num_classes is None:
        num_classes = labels.max().item() + 1
    
    # 创建 one-hot 编码张量
    one_hot = torch.zeros(b, n, num_classes, device=labels.device, dtype=torch.float32)
    
    # 使用 scatter_ 函数将对应位置设置为 1
    # 注意: scatter_ 要求索引与目标张量维度匹配，所以需要扩展 labels 的维度
    one_hot.scatter_(2, labels.unsqueeze(-1), 1)
    
    return one_hot

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        # h = 1.0 / (x.size()[1] - 1.0)

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1) #(h**(self.d/self.p))*

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class MultipleLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        
        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)
    
    def compute_loss(self, x, y, sep=True, **kwargs):
        num_feature = x.size(2)
        loss_list = []
        for i in range(num_feature):
            loss_list.append(self.lp_loss(x[:, :, i], y[:, :, i]))
        
        all_loss = sum(loss_list)
        
        if sep:
            return [all_loss] + loss_list
        else:
            return all_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)


def get_least_loaded_gpu():
    """
    获取当前占用率最低的GPU设备ID
    """
    try:
        # 初始化NVML
        pynvml.nvmlInit()

        # 获取GPU数量
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            print("未找到可用的NVIDIA GPU")
            return -1

        # 获取每个GPU的内存使用情况
        gpu_usage = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # 计算内存使用率和GPU利用率
            mem_usage = (mem_info.used / mem_info.total) * 100
            gpu_util = utilization.gpu

            # 综合负载指标（可以调整权重）
            total_usage = 0.7 * mem_usage + 0.3 * gpu_util
            gpu_usage.append((i, total_usage))

            print(f"GPU {i}: 内存使用率: {mem_usage:.2f}%, GPU利用率: {gpu_util}%, 综合负载: {total_usage:.2f}%")

        # 按综合负载排序，选择负载最低的GPU
        gpu_usage.sort(key=lambda x: x[1])
        best_gpu = gpu_usage[0][0]
        print(f"选择GPU {best_gpu}，负载: {gpu_usage[0][1]:.2f}%")

        return best_gpu

    except NVMLError as err:
        print(f"NVML错误: {err}")
        return -1
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def set_device():
    """
    设置设备：如果有GPU可用，选择占用率最低的GPU，否则使用CPU
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        return torch.device("cpu")

    best_gpu = get_least_loaded_gpu()

    if best_gpu == -1:
        print("使用CPU")
        return torch.device("cpu")
    else:
        print(f"使用GPU: {best_gpu}")
        return torch.device(f"cuda:{best_gpu}")#{best_gpu}


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

# Numpy things

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
