import os, re, glob, subprocess
from pathlib import Path

SRC_DIR = Path("/data/lichaofei/data/SCVCD-NEW-tratest")
DST_DIR = Path("/data/lichaofei/data/SCVCD-NEW-tratest_align8")

PIX_FMT = "yuv444p"   # 你的源是 yuv444p 就用这个；如果是 yuv420p 改成 yuv420p
FPS = 30              # raw yuv 其实不需要 fps，但写上无害

ALIGN = 8             # 对齐到 8 的倍数（满足 HM 的 minCU 约束）

DST_DIR.mkdir(parents=True, exist_ok=True)

def parse_wh(name):
    m = re.search(r"(\d+)x(\d+)\.yuv$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

for yuv in sorted(SRC_DIR.glob("*.yuv")):
    wh = parse_wh(yuv.name)
    if wh is None:
        print("[skip] cannot parse:", yuv.name)
        continue
    w, h = wh
    w2 = w - (w % ALIGN)
    h2 = h - (h % ALIGN)
    if w2 <= 0 or h2 <= 0:
        print("[skip] too small after align:", yuv.name)
        continue
    if w2 == w and h2 == h:
        # 尺寸已满足，直接复制（可选）
        out = DST_DIR / yuv.name
        out.write_bytes(yuv.read_bytes())
        print("[copy]", yuv.name)
        continue

    out_name = re.sub(r"\d+x\d+\.yuv$", f"{w2}x{h2}.yuv", yuv.name)
    out = DST_DIR / out_name

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", PIX_FMT,
        "-s", f"{w}x{h}",
        "-r", str(FPS),
        "-i", str(yuv),
        "-vf", f"crop={w2}:{h2}:0:0",
        "-pix_fmt", PIX_FMT,
        "-f", "rawvideo",
        str(out)
    ]
    subprocess.check_call(cmd)
    print("[crop]", yuv.name, "->", out.name)

print("DONE. output:", DST_DIR)