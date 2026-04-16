import os, json
from PIL import Image

SEQ_ROOT = "/data/lichaofei/data/SCVCD-NEW/test/sequences"
LIST     = "/data/lichaofei/data/SCVCD-NEW/test/DVC_test"
OUT_JSON = "dataset_config_scvcd_test_rgb.json"

FRAMES = 7
GOP    = 7

sequences = {}

with open(LIST, "r") as f:
    rels = [x.strip() for x in f.readlines() if x.strip()]

for rel in rels:
    im1 = os.path.join(SEQ_ROOT, rel, "im1.png")
    if not os.path.isfile(im1):
        raise FileNotFoundError(im1)
    w, h = Image.open(im1).size
    sequences[rel] = {"width": w, "height": h, "frames": FRAMES, "gop": GOP}

cfg = {
    "root_path": "/data/lichaofei/data/SCVCD-NEW/test",
    "test_classes": {
        "SCVCD_TEST": {
            "test": 1,
            "base_path": "sequences",
            "src_type": "png",
            "sequences": sequences
        }
    }
}

with open(OUT_JSON, "w") as f:
    json.dump(cfg, f, indent=2)

print("saved:", OUT_JSON)
print("num sequences:", len(sequences))