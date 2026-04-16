import json

IN_JSON  = "/home/chaofeili/DCVC-DC/dataset_config_scvcd_test_rgb.json"
OUT_JSON = "/home/chaofeili/DCVC-DC/dataset_config_scvcd_test_rgb_filtered.json"
OUT_DROP = "/home/chaofeili/DCVC-DC/dropped_sequences.txt"

MAX_SIDE = 2048
MAX_AREA = 2_500_000  # 像素数阈值

cfg = json.load(open(IN_JSON, "r"))
tc = cfg["test_classes"]["SCVCD_TEST"]
seqs = tc["sequences"]

kept = {}
dropped = []

for name, meta in seqs.items():
    w = int(meta["width"]); h = int(meta["height"])
    area = w * h
    if max(w, h) > MAX_SIDE or area > MAX_AREA:
        dropped.append((name, w, h, area))
    else:
        kept[name] = meta

# write outputs
tc["sequences"] = kept
json.dump(cfg, open(OUT_JSON, "w"), indent=2)

with open(OUT_DROP, "w") as f:
    for name, w, h, area in sorted(dropped, key=lambda x: x[3], reverse=True):
        f.write(f"{name} {w}x{h} area={area}\n")

print("Saved filtered config:", OUT_JSON)
print("Saved dropped list   :", OUT_DROP)
print("Kept:", len(kept), "Dropped:", len(dropped), "Total:", len(seqs))
if dropped:
    print("Largest dropped:", dropped[0])