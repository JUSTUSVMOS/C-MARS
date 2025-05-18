import lmdb
import pickle

env = lmdb.open('/mnt/c/code/c-mars/datasets/lmdb/ref-zom/train.lmdb', subdir=True, readonly=True)

with env.begin(write=False) as txn:
    keys = pickle.loads(txn.get(b'__keys__'))
    for idx in range(10):  # 檢查前10個
        k = keys[idx]
        ref = pickle.loads(txn.get(k))
        print(f"idx={idx}, seg_id={ref['seg_id']}, sents={ref['sents']}, num_sents={ref['num_sents']}, source={ref['source']}")

        # 你可以先只列出有哪些 key
