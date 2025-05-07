import os
import pickle
import random
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'ref-zom': {
        'train': 57624, 
        'val': 5762, 
        'test': 15406
    }
}
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s).

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize.
    context_length : int
        The context length to use; all CLIP models use 77 by default.
    truncate : bool
        Whether to truncate texts longer than context_length.

    Returns
    -------
    torch.LongTensor
        A tensor of shape (len(texts), context_length).
    """
    if isinstance(texts, str):
        texts = [texts]

    sot = _tokenizer.encoder.get("<|startoftext|>", 0)
    eot = _tokenizer.encoder.get("<|endoftext|>", 0)
    all_tokens = [[sot] + _tokenizer.encode(t) + [eot] for t in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
            else:
                raise RuntimeError(f"Input {texts[i]!r} too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    return result

def loads_pickle(buf):
    """Wrapper around pickle.loads."""
    return pickle.loads(buf)

class RefDataset(Dataset):
    """
    A generic RefCOCO / RefCOCO+ / G-Ref / Ref-ZOM dataset loader.

    Supports automatic decoding of LMDB-embedded masks for Ref-ZOM
    and uses random.choice to select full Python strings.
    """

    def __init__(self,
                 lmdb_dir: str,
                 mask_dir: str,
                 dataset: str,
                 split: str,
                 mode: str,
                 input_size: int,
                 word_length: int):
        super().__init__()
        self.lmdb_dir    = lmdb_dir
        self.mask_dir    = mask_dir
        self.dataset     = dataset.lower()
        self.split       = split
        self.mode        = mode           # 'train', 'val' or 'test'
        self.input_size  = (input_size, input_size)
        self.word_length = word_length

        # CLIP image normalization parameters
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3,1,1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3,1,1)

        # open LMDB
        self.env = lmdb.open(
            self.lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.keys   = loads_pickle(txn.get(b'__keys__'))
            self.length = len(self.keys)

        print(f"[RefDataset] dataset={self.dataset}, split={self.split}, length={self.length}")

        # for Ref-ZOM, if no mask_dir provided, use a cache folder
        if self.dataset == 'ref-zom' and (not self.mask_dir or self.mask_dir.strip()==''):
            cache = os.path.join(os.getcwd(), '_cache', 'masks-refzom')
            os.makedirs(cache, exist_ok=True)
            self.mask_dir = cache

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        try:
            if index >= self.length:
                raise IndexError(f"{index} out of range for dataset length {self.length}")

            # fetch record
            with self.env.begin(write=False) as txn:
                ref = loads_pickle(txn.get(self.keys[index]))

            # decode image
            ori = cv2.imdecode(
                np.frombuffer(ref['img'], dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if ori is None:
                raise ValueError(f"Failed to decode image at index {index}")
            img = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # load or decode mask
            seg_id    = ref.get('seg_id', index)
            mask_path = os.path.join(self.mask_dir, f"{seg_id}.png")
            if 'mask' in ref and ref['mask'] is not None:
                m = cv2.imdecode(
                    np.frombuffer(ref['mask'], dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                if self.mode != 'train' and not os.path.exists(mask_path):
                    cv2.imwrite(mask_path, m)
                mask = m
            else:
                if not os.path.exists(mask_path):
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Failed to read mask file at {mask_path}")

            # randomly select one sentence (guaranteed to be a Python str)
            sents = ref.get('sents', [])
            if not sents:
                raise ValueError(f"No sentences for index {index}")
            sent = str(random.choice(sents))

            # affine resize and pad
            mat, mat_inv = self.getTransformMat((h, w))
            border_val = [
                float(self.mean[0].item() * 255),
                float(self.mean[1].item() * 255),
                float(self.mean[2].item() * 255)
            ]
            img  = cv2.warpAffine(img,  mat, self.input_size,
                                  flags=cv2.INTER_CUBIC, borderValue=border_val)
            mask = cv2.warpAffine(mask, mat, self.input_size,
                                  flags=cv2.INTER_NEAREST, borderValue=0) / 255.0

            # tokenize text
            tokens   = tokenize(sent, self.word_length, truncate=True)
            word_vec = tokens.squeeze(0)

            # convert to tensor and normalize
            img_t  = torch.from_numpy(img.transpose(2,0,1)).float()
            img_t.div_(255.).sub_(self.mean).div_(self.std)
            mask_t = torch.from_numpy(mask).float()

            if self.mode in ['val', 'test']:
                return img_t, word_vec, {
                    'mask_dir': mask_path,
                    'inverse' : mat_inv,
                    'ori_size': np.array((h, w))
                }
            else:
                return img_t, word_vec, mask_t

        except Exception as e:
            print(f"[ERROR] Failed to process sample {index}: {e}")
            return None

    def getTransformMat(self, size: tuple) -> tuple:
        """
        Compute affine transformation matrix (and its inverse)
        to resize and pad the image to input_size.
        """
        h, w    = size
        ih, iw  = self.input_size
        scale   = min(ih / h, iw / w)
        nh, nw  = int(h * scale), int(w * scale)
        dx, dy  = (iw - nw) / 2, (ih - nh) / 2

        src     = np.array([[0,0], [w,0], [0,h]], np.float32)
        dst     = np.array([[dx,dy], [dx+nw,dy], [dx,dy+nh]], np.float32)
        mat     = cv2.getAffineTransform(src, dst)
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.dataset}:{self.split},mode={self.mode},"
                f"size={self.input_size},wlen={self.word_length})")
