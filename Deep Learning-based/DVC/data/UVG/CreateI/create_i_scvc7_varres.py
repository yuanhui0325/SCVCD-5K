import os, json, subprocess, sys
from PIL import Image

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def main():
    # python3 create_i_scvc7_varres.py SEQ_ROOT LIST CRF REC_ROOT [FPS]
    SEQ_ROOT = sys.argv[1]
    LIST     = sys.argv[2]
    CRF      = int(sys.argv[3])
    REC_ROOT = sys.argv[4]
    FPS      = int(sys.argv[5]) if len(sys.argv) > 5 else 30  # 这里其实用不到了

    os.makedirs(REC_ROOT, exist_ok=True)
    out_json = os.path.join(REC_ROOT, f"bpp_H265L{CRF}.json")
    bpp_map = {}

    # ===== 你想尽可能提高 I 帧质量，主要改这里 =====
    PRESET = "veryslow"     # 更慢更强压缩（同质量更省码率 / 同码率更高质量）
    TUNE_PSNR = True        # 追求 PSNR（可关）
    USE_444 = True          # 尝试 yuv444p（如果报错就改 False）
    # ===========================================

    with open(LIST, "r") as f:
        rel_folders = [x.strip() for x in f.readlines() if x.strip()]

    for rel in rel_folders:
        src_folder = os.path.join(SEQ_ROOT, rel)
        im1 = os.path.join(src_folder, "im1.png")
        if not os.path.isfile(im1):
            raise FileNotFoundError(im1)

        W, H = Image.open(im1).size
        cw = (W // 64) * 64
        ch = (H // 64) * 64
        if cw <= 0 or ch <= 0:
            raise RuntimeError(f"{rel}: invalid crop {W}x{H} -> {cw}x{ch}")

        out_dir = os.path.join(REC_ROOT, rel, f"H265L{CRF}")
        os.makedirs(out_dir, exist_ok=True)

        bitstream = os.path.join(out_dir, "I.hevc")
        rec_I = os.path.join(out_dir, "im0001.png")

        pix_fmt = "yuv444p" if USE_444 else "yuv420p"

        # x265 params
        x265_params = [
            f"crf={CRF}",
            "keyint=1", "min-keyint=1",
            "scenecut=0",
            "open-gop=0",
            "bframes=0",
        ]
        if TUNE_PSNR:
            x265_params.append("tune=psnr")
        if USE_444:
            # 如果你 x265 不支持 main444，会在这里或编码时报错 -> 把 USE_444=False
            x265_params.insert(0, "profile=main444-8")

        x265_params = ":".join(x265_params)

        # 1) encode ONLY ONE FRAME as I bitstream
        encode_cmd = (
            f'ffmpeg -y -loglevel error '
            f'-i "{im1}" -frames:v 1 '
            f'-vf "crop={cw}:{ch}:0:0" -pix_fmt {pix_fmt} '
            f'-c:v libx265 -preset {PRESET} '
            f'-x265-params "{x265_params}" '
            f'-f hevc "{bitstream}"'
        )
        run(encode_cmd)

        # 2) decode reconstructed I
        decode_cmd = f'ffmpeg -y -loglevel error -f hevc -i "{bitstream}" -frames:v 1 "{rec_I}"'
        run(decode_cmd)

        # 3) bpp from bitstream file size
        bytes_ = os.path.getsize(bitstream)
        bpp = (bytes_ * 8.0) / (cw * ch)

        bpp_map[rel] = {"bpp": bpp, "w": cw, "h": ch}
        print(f"{rel} -> {cw}x{ch}  I-bpp={bpp}")

    with open(out_json, "w") as f:
        json.dump(bpp_map, f, indent=2)
    print("saved:", out_json)

if __name__ == "__main__":
    main()