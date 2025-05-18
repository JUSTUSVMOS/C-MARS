import os
import pickle
import random
from typing import List, Union

import cv2
import lmdb
import numpy as np
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
        'val': 15406
    }
}
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Tokenize input string(s) into CLIP-compatible token ids.

    Args:
        texts (Union[str, List[str]]): Single string or list of strings.
        context_length (int): Max token sequence length.
        truncate (bool): If True, truncate sequences longer than context_length.

    Returns:
        torch.LongTensor: Token ids, shape [len(texts), context_length].
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
    return pickle.loads(buf)

class RefDataset(Dataset):
    """
    Universal Dataset:
    For ref-zom, automatically flatten samples so each entry is (image, one caption, one mask).
    For other datasets, one key per sample with a randomly selected caption per access.
    """

    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size, word_length):
        super().__init__()
        self.lmdb_dir    = lmdb_dir
        self.mask_dir    = mask_dir
        self.dataset     = dataset.lower()
        self.split       = split
        self.mode        = mode
        self.input_size  = (input_size, input_size)
        self.word_length = word_length

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3,1,1)
        self.std  = torch.tensor([0.26862954, 0.26130254, 0.27577711]).reshape(3,1,1)

        self.env = lmdb.open(
            self.lmdb_dir,
            subdir=os.path.isdir(self.lmdb_dir),
            readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            raw_keys = loads_pickle(txn.get(b'__keys__'))

        # For ref-zom: flatten so each (image, caption, mask) is a separate sample
        if self.dataset == 'ref-zom':
            cache_path = os.path.join(self.lmdb_dir, 'entries_flattened.pkl')
            if os.path.isfile(cache_path):
                with open(cache_path, 'rb') as f:
                    self.entries = pickle.load(f)
            else:
                self.entries = []
                with self.env.begin(write=False) as txn:
                    for key in raw_keys:
                        ref = loads_pickle(txn.get(key))
                        sents = ref.get('sents', [])
                        for i in range(len(sents)):
                            self.entries.append((key, i))
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.entries, f)
        else:
            # Other datasets: each key, caption randomly chosen at each __getitem__
            self.entries = [(key, None) for key in raw_keys]

        self.length = len(self.entries)
        print(f"[RefDataset] dataset={self.dataset}, split={self.split}, samples={self.length}")

        # Auto-generate mask cache dir for ref-zom if not provided
        if self.dataset == 'ref-zom' and not self.mask_dir:
            cache = os.path.join(os.getcwd(), '_cache', 'masks-refzom')
            os.makedirs(cache, exist_ok=True)
            self.mask_dir = cache

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        key, sent_idx = self.entries[index]
        with self.env.begin(write=False) as txn:
            ref = loads_pickle(txn.get(key))

        ori = cv2.imdecode(np.frombuffer(ref['img'], dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        sents = ref.get('sents', [])
        if not isinstance(sents, list):
            sents = [sents]

        # For ref-zom: flatten so every entry has sent_idx, others random sample a caption
        if sent_idx is not None:
            i = sent_idx
        else:
            i = random.randint(0, len(sents)-1)

        sent = str(sents[i])

        # Robust seg_id handling for multiple captions
        raw_id = ref.get('seg_id')
        if isinstance(raw_id, (list, tuple)) and len(raw_id) == len(sents):
            seg_ids = [int(x) for x in raw_id]
        elif raw_id is not None and not isinstance(raw_id, (list, tuple)):
            seg_ids = [int(raw_id)] * len(sents)
        else:
            seg_ids = [0] * len(sents)
        seg_id = seg_ids[i]

        mask_raw = cv2.imdecode(np.frombuffer(ref['mask'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mat, mat_inv = self.getTransformMat((h, w))
        border = [float(self.mean[k].item()*255) for k in range(3)]
        img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC, borderValue=border)
        mask = cv2.warpAffine(mask_raw, mat, self.input_size, flags=cv2.INTER_NEAREST, borderValue=0) / 255.0
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.).sub(self.mean).div(self.std)
        mask_t = torch.from_numpy(mask).float()
        tokens = tokenize(sent, self.word_length, truncate=True)
        word_vec = tokens.squeeze(0)

        # Return based on mode
        if self.mode == 'train':
            return img_t, word_vec, mask_t
        elif self.mode == 'val':
            return img_t, word_vec, mask_t, {
                'inverse': mat_inv,
                'ori_size': (h, w),
                'source_type': ref.get('source', 'unknown') if self.dataset == 'ref-zom' else 'normal',
                'seg_id': seg_id,
            }
        else:  # test
            return img_t, word_vec, mask_t, {
                'inverse': mat_inv,
                'ori_size': (h, w),
                'sents': [sent],
                'source_type': [ref.get('source', 'unknown')] if self.dataset == 'ref-zom' else ['normal'],
                'seg_id': [seg_id],
            }

    def getTransformMat(self, size: tuple):
        """
        Compute affine transform and its inverse for resizing image/mask to self.input_size.
        """
        h, w = size
        ih, iw = self.input_size
        scale = min(ih/h, iw/w)
        nh, nw = int(h*scale), int(w*scale)
        dx, dy = (iw-nw)/2, (ih-nh)/2
        src = np.array([[0,0],[w,0],[0,h]], np.float32)
        dst = np.array([[dx,dy],[dx+nw,dy],[dx,dy+nh]], np.float32)
        mat     = cv2.getAffineTransform(src, dst)
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.dataset}:{self.split},mode={self.mode},"
                f"size={self.input_size},wlen={self.word_length})")
