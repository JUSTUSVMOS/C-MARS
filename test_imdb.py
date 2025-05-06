import lmdb
import pyarrow as pa
import logging

def verify_lmdb(lmdb_dir):
    env = lmdb.open(lmdb_dir, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        try:
            length = pa.deserialize(txn.get(b'__len__'))
            keys = pa.deserialize(txn.get(b'__keys__'))
            logging.info(f"Dataset length: {length}")
            logging.info(f"Number of keys: {len(keys)}")

            for key in keys:
                data = txn.get(key)
                if data is None:
                    logging.error(f"Data for key {key} is missing")
                    return False
                pa.deserialize(data)
        except Exception as e:
            logging.error(f"Error while verifying LMDB data: {e}")
            return False
    return True

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    lmdb_dir = '/mnt/d/code/cris.pytorch-master/datasets/lmdb/refcoco/val.lmdb'  # 替换为你的lmdb路径
    if verify_lmdb(lmdb_dir):
        logging.info("LMDB data verification passed.")
    else:
        logging.error("LMDB data verification failed.")
