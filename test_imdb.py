import os, pickle, lmdb, cv2, numpy as np, tqdm
from pathlib import Path
from pycocotools.coco import COCO

# ========= 1. 基本路徑 =========
PICKLE_PATH   = 'datasets/ref-zom/ref_zom.p'          # Ref-ZOM .p
COCO_ANN_JSON = 'datasets/ref-zom/instances.json'     # 唯一 JSON
COCO_IMG_ROOT = Path('datasets/images/trainval2014')  # 所有 JPG 平鋪於此
LMDB_DIR      = Path('datasets/lmdb/ref-zom')
MAP_SIZE_GB   = 80
# =================================

# ---------- 2. 載入資料 ----------
print('loading Ref-ZOM pickle ...')
items = pickle.load(open(PICKLE_PATH, 'rb'))

print('loading COCO json ...')
coco = COCO(COCO_ANN_JSON)

# ---------- 3. 開兩個 LMDB ----------
lmdb_env = {
    'train': lmdb.open(str(LMDB_DIR / 'train.lmdb'),
                       map_size=MAP_SIZE_GB << 30, subdir=True),
    'val'  : lmdb.open(str(LMDB_DIR / 'val.lmdb'),
                       map_size=MAP_SIZE_GB << 30, subdir=True),
}
keys_cache = {'train': [], 'val': []}

# ---------- 4. 文字欄位自動偵測 ----------
TEXT_KEY = None
for cand in ('sent', 'sentence', 'sentences', 'expression', 'caption'):
    if cand in items[0]:
        TEXT_KEY = cand
        break
if TEXT_KEY is None:
    raise RuntimeError(f'cannot find sentence field in pickle keys={items[0].keys()}')

print(f'[*] use "{TEXT_KEY}" as sentence field')  # <<<

# ---------- 5. 小工具 ----------
def img_path_from_file_name(name: str) -> Path:
    return COCO_IMG_ROOT / name                     # 同層尋找 JPG

def encode_img(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'img missing: {path}')
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes(), img.shape[:2]             # (H, W)

def encode_mask(ann_ids, size):
    if not isinstance(ann_ids, (list, tuple)):
        ann_ids = [ann_ids]

    masks = []
    for aid in ann_ids:
        ann = coco.loadAnns([int(aid)])[0]
        if len(ann.get('segmentation', [])) == 0:
            continue
        m = coco.annToMask(ann)
        if m is None or m.sum() == 0:
            continue
        masks.append(m)

    if not masks:                                   # 無有效 mask → 全 0
        mask = np.zeros(size, np.uint8)
    else:
        mask = np.any(np.stack(masks, 0), 0).astype(np.uint8) * 255

    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    _, buf = cv2.imencode('.png', mask)
    return buf.tobytes()

# ---------- 6. 寫入 LMDB ----------
for split_tag in ('train', 'test'):                 # Ref-ZOM 的 test 當 val
    split_key = 'train' if split_tag == 'train' else 'val'
    env       = lmdb_env[split_key]

    subset = [it for it in items if it['split'] == split_tag]
    with env.begin(write=True) as txn:
        for idx, it in enumerate(tqdm.tqdm(subset, desc=split_key)):
            img_path      = img_path_from_file_name(it['file_name'])
            img_bytes, wh = encode_img(img_path)
            mask_bytes    = encode_mask(it['ann_id'], wh)

            sent_list = it[TEXT_KEY]                # <<< 可能已是 list
            if not isinstance(sent_list, (list, tuple)):
                sent_list = [sent_list]

            ref = {
                'img'      : img_bytes,
                'mask'     : mask_bytes,
                'seg_id'   : it['ann_id'],
                'sents'    : sent_list,             # <<< 存整個句子列表
                'num_sents': len(sent_list),
            }
            key = f'{idx:08d}'.encode()
            txn.put(key, pickle.dumps(ref))
            keys_cache[split_key].append(key)

        txn.put(b'__keys__', pickle.dumps(keys_cache[split_key]))
    env.sync(); env.close()

print('✔  Ref-ZOM LMDB 建立完成 →', LMDB_DIR)