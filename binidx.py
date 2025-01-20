from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate

# 定义数据类型映射表
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

# 根据数据类型获取对应的编码
def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

# 获取索引文件路径
def index_file_path(prefix_path):
    return prefix_path + ".idx"

# 获取数据文件路径
def data_file_path(prefix_path):
    return prefix_path + ".bin"

# 内存映射索引数据集类
class MMapIndexedDataset(torch.utils.data.Dataset):
    # 索引类，用于管理数据索引
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"  # 文件头魔数

        @classmethod
        def writer(cls, path, dtype):
            # 索引写入器类
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    # 写入魔数字符串用于文件格式校验
                    self._file.write(cls._HDR_MAGIC)
                    # 写入版本号 (64位无符号整数，小端序)
                    self._file.write(struct.pack("<Q", 1))
                    # 写入数据类型编码 (8位无符号整数)
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    # 计算数据指针位置
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    # 写入索引数据
                    pointers = self._get_pointers(sizes)

                    # 写入样本数量 (64位无符号整数)
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # 写入文档数量 (64位无符号整数)
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    # 写入样本大小数组
                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    # 写入指针数组
                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    # 写入文档索引数组
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()
        
        def __init__(self, path, skip_warmup=False):
            # 初始化索引
            with open(path, "rb") as stream:
                # 校验文件头
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "索引文件格式不匹配，请检查--dataset-impl配置"
                )
                # 读取版本号
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                # 读取数据类型编码
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                # 读取样本数量和文档数量
                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            # 创建内存映射
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            # 读取样本大小数组
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            # 读取指针数组
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            # 读取文档索引数组
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            # 清理内存映射
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            # 获取指定索引的指针和大小
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        # 初始化数据集
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        # 执行初始化操作
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        # 清理资源
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        # 获取指定索引的数据
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("索引切片必须是连续的")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """获取数据集中的单个项目，可选择只返回部分数据
        
        get(idx) 与 [idx] 相同，但 get() 不支持切片
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        # 检查索引文件和数据文件是否存在
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
