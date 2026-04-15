import os
import glob
import subprocess
import re
import csv
import math
import numpy as np
import torch
from pytorch_msssim import ms_ssim

# ==================== 配置区域 ====================

HM_EXEC = "/home/chaofeili/Tradition/HM-16.20+SCM-8.8/bin/TAppEncoderStatic"
CONFIG_FILE = "/home/chaofeili/Tradition/HM-16.20+SCM-8.8/cfg/encoder_randomaccess_main_scc.cfg"

# 建议用已经对齐到 8/16 倍数的 yuv 目录
DATASET_DIR = "/data/lichaofei/data/SCVCD-NEW-tratest_align8"
OUTPUT_DIR  = "/data/lichaofei/data/HEVC-SCC_results"

QP_LIST = [22, 27, 32, 37, 42]

INPUT_PIX_FMT = "yuv444p"   # yuv444p 或 yuv420p
INPUT_BITDEPTH = 8          # 源位深（一般 8）

# HM-SCC 输出 recon 位深：很多配置内部 10bit，但也可能写 8bit。
# 如果你确认 recon 是 8bit，请改成 8
REC_BITDEPTH = 10

FPS_DEFAULT = 30

# 是否计算 RGB 指标（非常慢）
DO_RGB = False

# ===================================================

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    res_match = re.search(r'(\d+)x(\d+)', filename)
    fps_match = re.search(r'_(\d+)\.yuv$', filename)
    if not res_match:
        print(f"[ERR] 文件名 {filename} 未包含分辨率 (例如 1920x1080)")
        return None, None, None
    w = int(res_match.group(1))
    h = int(res_match.group(2))
    fps = int(fps_match.group(1)) if fps_match else FPS_DEFAULT
    return w, h, fps

def bytes_per_frame(w, h, pix_fmt, bitdepth):
    bps = 2 if bitdepth > 8 else 1
    if "444" in pix_fmt:
        return w * h * 3 * bps
    else:
        return int(w * h * 3 / 2) * bps

def read_y_plane(fp, w, h, pix_fmt, bitdepth):
    """读取一帧 Y 平面，并跳过 UV（支持 420/444，8/10bit(16bit存储)）"""
    bps = 2 if bitdepth > 8 else 1
    y_bytes = w * h * bps
    buf = fp.read(y_bytes)
    if len(buf) != y_bytes:
        return None

    if bitdepth > 8:
        y = np.frombuffer(buf, dtype=np.uint16).reshape(h, w)
    else:
        y = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)

    # skip UV
    if "444" in pix_fmt:
        skip = w * h * 2 * bps
    else:
        skip = int(w * h / 2) * bps
    fp.seek(skip, os.SEEK_CUR)
    return y

@torch.no_grad()
def calc_y_metrics_with_sse(w, h, nframes, src_yuv, rec_yuv, pix_fmt, src_bitdepth, rec_bitdepth, device):
    """
    返回：
      avg_y_psnr, avg_y_msssim,
      sse_sum, frames_cnt, pixels_cnt
    其中 dataset 的 Y-PSNR 用 sse_sum/pixels_cnt 算全局 MSE 更标准。
    """
    max_src = (1 << src_bitdepth) - 1
    shift = max(0, rec_bitdepth - src_bitdepth)

    sum_psnr = 0.0
    sum_msssim = 0.0
    sse_sum = 0.0
    cnt = 0
    pixels_per_frame = w * h

    with open(src_yuv, "rb") as fs, open(rec_yuv, "rb") as fr:
        for _ in range(nframes):
            y0 = read_y_plane(fs, w, h, pix_fmt, src_bitdepth)
            y1 = read_y_plane(fr, w, h, pix_fmt, rec_bitdepth)
            if y0 is None or y1 is None:
                break

            # bitdepth 对齐：例如 rec=10 -> src=8 右移2
            if shift > 0:
                y1 = (y1 >> shift)

            # 统一转 float32（避免 torch 不支持 uint16）
            y0_f = y0.astype(np.float32)
            y1_f = y1.astype(np.float32)

            # SSE / PSNR
            diff = y0_f - y1_f
            sse = float(np.sum(diff * diff))
            mse = sse / pixels_per_frame
            sse_sum += sse

            if mse < 1e-12:
                psnr = 100.0
            else:
                psnr = 10.0 * math.log10((max_src * max_src) / mse)
            sum_psnr += psnr

            # MS-SSIM(Y)：归一化到 0..1
            t0 = torch.from_numpy(y0_f.copy()).unsqueeze(0).unsqueeze(0).to(device) / max_src
            t1 = torch.from_numpy(y1_f.copy()).unsqueeze(0).unsqueeze(0).to(device) / max_src
            sum_msssim += float(ms_ssim(t0, t1, data_range=1.0, size_average=True).item())

            cnt += 1

    if cnt == 0:
        return 0.0, 0.0, 0.0, 0, 0
    pixels_cnt = pixels_per_frame * cnt
    return (sum_psnr / cnt), (sum_msssim / cnt), sse_sum, cnt, pixels_cnt

@torch.no_grad()
def calc_rgb_metrics_ffmpeg(w, h, nframes, src_yuv, rec_yuv, device, input_pix_fmt, src_bitdepth, rec_bitdepth):
    """
    （可选）通过 ffmpeg 转 RGB24，计算 RGB-PSNR & RGB-MS-SSIM（非常慢）
    """
    frame_size = w * h * 3

    src_fmt = input_pix_fmt
    if src_bitdepth > 8 and "10" not in src_fmt:
        src_fmt = src_fmt + "10le"

    rec_fmt = input_pix_fmt
    if rec_bitdepth > 8 and "10" not in rec_fmt:
        rec_fmt = rec_fmt + "10le"

    cmd_src = [
        "ffmpeg", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", src_fmt, "-s", f"{w}x{h}", "-i", src_yuv,
        "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"
    ]
    cmd_rec = [
        "ffmpeg", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", rec_fmt, "-s", f"{w}x{h}", "-i", rec_yuv,
        "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"
    ]

    p0 = subprocess.Popen(cmd_src, stdout=subprocess.PIPE)
    p1 = subprocess.Popen(cmd_rec, stdout=subprocess.PIPE)

    sum_psnr = 0.0
    sum_msssim = 0.0
    cnt = 0

    try:
        for _ in range(nframes):
            a = p0.stdout.read(frame_size)
            b = p1.stdout.read(frame_size)
            if not a or not b:
                break
            x = np.frombuffer(a, dtype=np.uint8).reshape(h, w, 3)
            y = np.frombuffer(b, dtype=np.uint8).reshape(h, w, 3)

            mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
            psnr = 100.0 if mse < 1e-12 else 10.0 * math.log10((255.0 * 255.0) / mse)
            sum_psnr += psnr

            tx = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
            ty = torch.from_numpy(y.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
            sum_msssim += float(ms_ssim(tx, ty, data_range=1.0, size_average=True).item())

            cnt += 1
    finally:
        if p0.stdout: p0.stdout.close()
        if p1.stdout: p1.stdout.close()
        p0.terminate()
        p1.terminate()

    if cnt == 0:
        return 0.0, 0.0, 0
    return sum_psnr / cnt, sum_msssim / cnt, cnt

def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_video_csv = os.path.join(OUTPUT_DIR, "results_hevc_scc_per_video.csv")
    dataset_csv   = os.path.join(OUTPUT_DIR, "results_hevc_scc_dataset.csv")

    yuv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.yuv")))
    print(f"找到 {len(yuv_files)} 个 YUV 文件...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备计算 MS-SSIM: {device}")
    print(f"DO_RGB={DO_RGB} (RGB 指标会非常慢)")

    # dataset 汇总（按 QP）
    dataset_stat = {
        qp: {
            "bits": 0,
            "pixels": 0,
            "sse": 0.0,
            "msssim_sum": 0.0,
            "frames": 0,
            "rgb_psnr_sum": 0.0,
            "rgb_msssim_sum": 0.0,
            "rgb_frames": 0,
        } for qp in QP_LIST
    }

    with open(per_video_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Sequence", "QP",
            "BPP",
            "PSNR_Y_calc", "MS_SSIM_Y",
            "PSNR_RGB", "MS_SSIM_RGB",
            "Width", "Height", "Frames"
        ])

        for yuv_path in yuv_files:
            seq_name = os.path.basename(yuv_path)
            w, h, fps = parse_filename(yuv_path)
            if not w:
                continue

            fb = bytes_per_frame(w, h, INPUT_PIX_FMT, INPUT_BITDEPTH)
            frame_count = int(os.path.getsize(yuv_path) // fb)

            print(f"\n序列: {seq_name} | {w}x{h} | {frame_count} 帧")

            for qp in QP_LIST:
                bit_file = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}.bin")
                rec_yuv  = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}_rec.yuv")

                cmd = [
                    HM_EXEC,
                    "-c", CONFIG_FILE,
                    "-i", yuv_path,
                    "-b", bit_file,
                    "-o", rec_yuv,
                    "-wdt", str(w),
                    "-hgt", str(h),
                    "-fr", str(fps),
                    "-f", str(frame_count),
                    "-q", str(qp),
                    f"--InputBitDepth={INPUT_BITDEPTH}",
                    "--InternalBitDepth=10",
                    "--InputChromaFormat=444" if "444" in INPUT_PIX_FMT else "--InputChromaFormat=420",
                ]

                ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if ret.returncode != 0:
                    print(f"Error: 编码失败 QP {qp}, returncode={ret.returncode}")
                    print("CMD:", " ".join(cmd))
                    print("STDOUT:", ret.stdout[:1200])
                    print("STDERR:", ret.stderr[:1200])
                    continue

                if not os.path.exists(bit_file) or not os.path.exists(rec_yuv):
                    print(f"Error: output not found QP {qp}")
                    continue

                bits = os.path.getsize(bit_file) * 8
                bpp = bits / (w * h * frame_count)

                # Y metrics + SSE
                y_psnr, y_msssim, sse_sum, y_frames, y_pixels = calc_y_metrics_with_sse(
                    w, h, frame_count, yuv_path, rec_yuv,
                    INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH, device
                )

                # RGB metrics (optional)
                if DO_RGB:
                    rgb_psnr, rgb_msssim, rgb_frames = calc_rgb_metrics_ffmpeg(
                        w, h, frame_count, yuv_path, rec_yuv,
                        device, INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH
                    )
                    print(f"  QP{qp}: bpp={bpp:.4f}, Y-PSNR={y_psnr:.3f}, Y-MS-SSIM={y_msssim:.4f}, RGB-PSNR={rgb_psnr:.3f}, RGB-MS-SSIM={rgb_msssim:.4f}")
                else:
                    rgb_psnr, rgb_msssim, rgb_frames = "", "", 0
                    print(f"  QP{qp}: bpp={bpp:.4f}, Y-PSNR={y_psnr:.3f}, Y-MS-SSIM={y_msssim:.4f}")

                # per-video csv
                writer.writerow([
                    seq_name, qp,
                    f"{bpp:.6f}",
                    f"{y_psnr:.4f}", f"{y_msssim:.6f}",
                    f"{rgb_psnr}" if rgb_psnr != "" else "",
                    f"{rgb_msssim}" if rgb_msssim != "" else "",
                    w, h, y_frames
                ])

                # dataset accumulate
                st = dataset_stat[qp]
                st["bits"] += bits
                st["pixels"] += y_pixels
                st["sse"] += sse_sum
                st["msssim_sum"] += y_msssim * y_frames
                st["frames"] += y_frames

                if DO_RGB and rgb_frames > 0:
                    st["rgb_psnr_sum"] += rgb_psnr * rgb_frames
                    st["rgb_msssim_sum"] += rgb_msssim * rgb_frames
                    st["rgb_frames"] += rgb_frames

                # 调试期建议保留；稳定后可删除节省空间
                # os.remove(bit_file)
                # os.remove(rec_yuv)

    # dataset summary
    with open(dataset_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "QP",
            "Dataset_BPP",
            "Dataset_Y_PSNR", "Dataset_Y_MS_SSIM",
            "Dataset_RGB_PSNR", "Dataset_RGB_MS_SSIM",
            "Total_Frames", "Total_Pixels"
        ])

        print("\n========== DATASET SUMMARY (per QP) ==========")
        for qp in QP_LIST:
            st = dataset_stat[qp]
            if st["pixels"] == 0 or st["frames"] == 0:
                continue

            dataset_bpp = st["bits"] / st["pixels"]

            maxv = (1 << INPUT_BITDEPTH) - 1
            global_mse = st["sse"] / st["pixels"]
            dataset_y_psnr = 100.0 if global_mse < 1e-12 else 10.0 * math.log10((maxv * maxv) / global_mse)
            dataset_y_msssim = st["msssim_sum"] / st["frames"]

            if DO_RGB and st["rgb_frames"] > 0:
                dataset_rgb_psnr = st["rgb_psnr_sum"] / st["rgb_frames"]
                dataset_rgb_msssim = st["rgb_msssim_sum"] / st["rgb_frames"]
            else:
                dataset_rgb_psnr = ""
                dataset_rgb_msssim = ""

            print(f"QP{qp}: bpp={dataset_bpp:.6f}, Y-PSNR={dataset_y_psnr:.4f}, Y-MS-SSIM={dataset_y_msssim:.6f}")

            writer.writerow([
                qp,
                f"{dataset_bpp:.6f}",
                f"{dataset_y_psnr:.4f}", f"{dataset_y_msssim:.6f}",
                f"{dataset_rgb_psnr}" if dataset_rgb_psnr != "" else "",
                f"{dataset_rgb_msssim}" if dataset_rgb_msssim != "" else "",
                st["frames"], st["pixels"]
            ])

    print("\n完成：")
    print("  per-video results :", per_video_csv)
    print("  dataset summary   :", dataset_csv)

if __name__ == "__main__":
    run_test()