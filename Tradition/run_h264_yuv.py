import os
import glob
import subprocess
import re
import csv
import math
import numpy as np
import torch
from pytorch_msssim import ms_ssim

# ==================== 配置区域 (请修改) ====================

# 1. JM 编码器路径 (lencod)
# 您的路径是: /home/chaofeili/Tradition/JM-19.0/bin/lencod.exe
JM_EXEC = "/home/chaofeili/Tradition/JM-19.0/bin/lencod.exe"

# 2. 配置文件 (JM 必须指定一个基础 cfg)
# 建议使用 encoder_main.cfg 或者 encoder.cfg
CONFIG_FILE = "/home/chaofeili/Tradition/JM-19.0/bin/encoder_main.cfg"

# 3. YUV 数据集路径
DATASET_DIR = "/data/lichaofei/data/SCVCD-NEW-tratest"

# 4. 结果保存路径
OUTPUT_DIR = "/data/lichaofei/data/AVC_results"

# 5. QP 设置 (H.264 标准测试点)
QP_LIST = [22, 27, 32, 37, 42]

# 6. YUV 格式设置
INPUT_PIX_FMT = "yuv444p"   # yuv444p 或 yuv420p
INPUT_BITDEPTH = 8          # 8 或 10

# JM 的输出位深通常与输入一致。如果是 8bit 输入，这里填 8。
REC_BITDEPTH = 8

FPS_DEFAULT = 30

# 是否计算 RGB 指标
DO_RGB = False

# ==========================================================

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    res_match = re.search(r'(\d+)x(\d+)', filename)
    fps_match = re.search(r'_(\d+)\.yuv$', filename)
    if not res_match:
        print(f"[ERR] 文件名 {filename} 未包含分辨率")
        return None, None, None
    w = int(res_match.group(1))
    h = int(res_match.group(2))
    fps = int(fps_match.group(1)) if fps_match else FPS_DEFAULT
    return w, h, fps

def bytes_per_frame(w, h, pix_fmt, bitdepth):
    bps = 2 if bitdepth > 8 else 1
    if "444" in pix_fmt: return w * h * 3 * bps
    else: return int(w * h * 3 / 2) * bps

def read_y_plane(fp, w, h, pix_fmt, bitdepth):
    bps = 2 if bitdepth > 8 else 1
    y_bytes = w * h * bps
    y_buf = fp.read(y_bytes)
    if len(y_buf) != y_bytes: return None
    if bitdepth > 8: y = np.frombuffer(y_buf, dtype=np.uint16).reshape(h, w)
    else: y = np.frombuffer(y_buf, dtype=np.uint8).reshape(h, w)
    # skip UV
    if "444" in pix_fmt: skip = w * h * 2 * bps
    else: skip = int(w * h / 2) * bps
    fp.seek(skip, os.SEEK_CUR)
    return y

@torch.no_grad()
def calc_y_metrics(w, h, nframes, src_yuv, rec_yuv, pix_fmt, src_bitdepth, rec_bitdepth, device):
    """
    计算 Y-PSNR 和 Y-MS-SSIM
    已修复 uint16 报错问题
    """
    max_src = (1 << src_bitdepth) - 1
    shift = max(0, rec_bitdepth - src_bitdepth)
    sum_psnr = 0.0
    sum_msssim = 0.0
    cnt = 0

    with open(src_yuv, "rb") as fs, open(rec_yuv, "rb") as fr:
        for _ in range(nframes):
            y0 = read_y_plane(fs, w, h, pix_fmt, src_bitdepth)
            y1 = read_y_plane(fr, w, h, pix_fmt, rec_bitdepth)
            if y0 is None or y1 is None: break

            if shift > 0: y1 = (y1 >> shift)
            
            # 【关键修复】转 float32 避免 torch 报错
            y0_f = y0.astype(np.float32)
            y1_f = y1.astype(np.float32)

            diff = y0_f - y1_f
            mse = float(np.mean(diff * diff))
            psnr = 100.0 if mse < 1e-12 else 10.0 * math.log10((max_src**2) / mse)
            sum_psnr += psnr

            t0 = torch.from_numpy(y0_f).unsqueeze(0).unsqueeze(0).to(device) / max_src
            t1 = torch.from_numpy(y1_f).unsqueeze(0).unsqueeze(0).to(device) / max_src
            sum_msssim += float(ms_ssim(t0, t1, data_range=1.0, size_average=True).item())
            cnt += 1
            
    if cnt == 0: return 0.0, 0.0
    return sum_psnr / cnt, sum_msssim / cnt

@torch.no_grad()
def calc_rgb_metrics_ffmpeg(w, h, nframes, src_yuv, rec_yuv, device, input_pix_fmt, src_bitdepth, rec_bitdepth):
    frame_size = w * h * 3
    src_fmt = input_pix_fmt + ("10le" if src_bitdepth > 8 and "10" not in input_pix_fmt else "")
    rec_fmt = input_pix_fmt + ("10le" if rec_bitdepth > 8 and "10" not in input_pix_fmt else "")

    cmd_base = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-s", f"{w}x{h}"]
    cmd_src = cmd_base + ["-pix_fmt", src_fmt, "-i", src_yuv, "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    cmd_rec = cmd_base + ["-pix_fmt", rec_fmt, "-i", rec_yuv, "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]

    p0 = subprocess.Popen(cmd_src, stdout=subprocess.PIPE)
    p1 = subprocess.Popen(cmd_rec, stdout=subprocess.PIPE)
    sum_psnr = 0.0
    sum_msssim = 0.0
    cnt = 0
    try:
        for _ in range(nframes):
            a = p0.stdout.read(frame_size)
            b = p1.stdout.read(frame_size)
            if not a or not b: break
            
            x = np.frombuffer(a, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
            y = np.frombuffer(b, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
            
            mse = float(np.mean((x - y) ** 2))
            psnr = 100.0 if mse < 1e-12 else 10.0 * math.log10((255**2) / mse)
            sum_psnr += psnr
            
            tx = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(device) / 255.0
            ty = torch.from_numpy(y.transpose(2,0,1)).unsqueeze(0).float().to(device) / 255.0
            sum_msssim += float(ms_ssim(tx, ty, data_range=1.0, size_average=True).item())
            cnt += 1
    finally:
        p0.terminate(); p1.terminate()
        
    if cnt == 0: return 0.0, 0.0
    return sum_psnr / cnt, sum_msssim / cnt

def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "results_h264_jm.csv")
    
    # 汇总数据
    metrics_summary = {qp: {'bpp': [], 'psnr_y': [], 'msssim_y': [], 'psnr_rgb': [], 'msssim_rgb': []} for qp in QP_LIST}

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["Sequence", "QP", "BPP", "PSNR_Y_JMlog", "PSNR_Y_calc", "MS_SSIM_Y", "PSNR_RGB", "MS_SSIM_RGB", "Width", "Height", "Frames"])
        
        yuv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.yuv")))
        print(f"找到 {len(yuv_files)} 个 YUV 文件...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        for yuv_path in yuv_files:
            seq_name = os.path.basename(yuv_path)
            w, h, fps = parse_filename(yuv_path)
            if not w: continue
            
            fb = bytes_per_frame(w, h, INPUT_PIX_FMT, INPUT_BITDEPTH)
            frame_count = int(os.path.getsize(yuv_path) // fb)
            print(f"\n序列: {seq_name} | {w}x{h} | {frame_count} 帧")

            for qp in QP_LIST:
                bit_file = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}.264")
                rec_yuv  = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}_rec.yuv")
                
                # JM 参数构造 (注意参数名!)
                # -d: 指定配置文件
                # -p: 覆盖参数 (JM 使用 Parameter=Value 格式)
                cmd = [
                    JM_EXEC,
                    "-d", CONFIG_FILE,
                    "-p", f"InputFile={yuv_path}",
                    "-p", f"OutputFile={bit_file}",
                    "-p", f"ReconFile={rec_yuv}",
                    "-p", f"SourceWidth={w}",
                    "-p", f"SourceHeight={h}",
                    "-p", f"FrameRate={float(fps)}",
                    "-p", f"FramesToBeEncoded={frame_count}",
                    # QP 设置 (JM 需要分别设置 I/P/B 帧 QP)
                    "-p", f"QPISlice={qp}",
                    "-p", f"QPPSlice={qp}",
                    "-p", f"QPBSlice={qp}",
                    # 格式设置
                    "-p", f"YUVFormat={3 if '444' in INPUT_PIX_FMT else 1}", # 0=400, 1=420, 2=422, 3=444
                    "-p", "ProfileIDC=244", # High 4:4:4 Predictive Profile (重要!)
                    "-p", "IntraPeriod=32", # 类似 Random Access
                    "-p", "NumberBFrames=0", # DVC 对比通常用 Low Delay P (IPPP)
                    "-p", "LevelIDC=62",    # Level 6.2
                    "-p", "SearchRange=64", # 搜索范围
                ]
                
                # 运行编码
                ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if ret.returncode != 0:
                    print(f"Error QP {qp}: {ret.returncode}")
                    # JM 的报错有时候在 stdout 有时候在 stderr，都打印一点
                    print("STDOUT tail:", ret.stdout[-300:])
                    print("STDERR tail:", ret.stderr[-300:])
                    continue
                
                # 1. BPP
                if os.path.exists(bit_file):
                    total_bits = os.path.getsize(bit_file) * 8
                    bpp = total_bits / (w * h * frame_count)
                else: bpp = 0
                
                # 2. 解析 Log PSNR (JM 格式通常为 " SNR Y(dB) :  40.123")
                psnr_log = 0.0
                for line in ret.stdout.splitlines():
                    if "SNR Y(dB)" in line:
                        try: psnr_log = float(line.split(":")[1].strip())
                        except: pass

                # 3. 计算指标
                psnr_y, msssim_y = calc_y_metrics(w, h, frame_count, yuv_path, rec_yuv, INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH, device)
                
                psnr_rgb, msssim_rgb = 0.0, 0.0
                if DO_RGB:
                    psnr_rgb, msssim_rgb = calc_rgb_metrics_ffmpeg(w, h, frame_count, yuv_path, rec_yuv, device, INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH)
                    print(f"  QP{qp}: BPP={bpp:.4f}, Y-PSNR={psnr_y:.2f}, RGB-PSNR={psnr_rgb:.2f}")
                else:
                    print(f"  QP{qp}: BPP={bpp:.4f}, Y-PSNR={psnr_y:.2f}, Y-MS-SSIM={msssim_y:.4f}")

                # 收集汇总
                metrics_summary[qp]['bpp'].append(bpp)
                metrics_summary[qp]['psnr_y'].append(psnr_y)
                metrics_summary[qp]['msssim_y'].append(msssim_y)
                if DO_RGB:
                    metrics_summary[qp]['psnr_rgb'].append(psnr_rgb)
                    metrics_summary[qp]['msssim_rgb'].append(msssim_rgb)

                writer.writerow([seq_name, qp, f"{bpp:.6f}", f"{psnr_log:.4f}", f"{psnr_y:.4f}", f"{msssim_y:.6f}", f"{psnr_rgb}", f"{msssim_rgb}", w, h, frame_count])
                
                # 清理
                # if os.path.exists(bit_file): os.remove(bit_file)
                # if os.path.exists(rec_yuv): os.remove(rec_yuv)

        # === 写入平均值 ===
        writer.writerow([])
        writer.writerow(["=== DATASET AVERAGE ==="])
        writer.writerow(["QP", "Avg_BPP", "Avg_PSNR_Y", "Avg_MS_SSIM_Y", "Avg_PSNR_RGB", "Avg_MS_SSIM_RGB"])
        
        print("\n=== DATASET AVERAGE RESULTS ===")
        for qp in QP_LIST:
            data = metrics_summary[qp]
            if not data['bpp']: continue
            
            avg_bpp = np.mean(data['bpp'])
            avg_psnr = np.mean(data['psnr_y'])
            avg_ssim = np.mean(data['msssim_y'])
            
            avg_psnr_rgb = np.mean(data['psnr_rgb']) if DO_RGB else 0
            avg_ssim_rgb = np.mean(data['msssim_rgb']) if DO_RGB else 0
            
            print(f"QP {qp}: BPP={avg_bpp:.4f} | Y-PSNR={avg_psnr:.2f} | Y-MS-SSIM={avg_ssim:.4f}")
            
            writer.writerow([qp, f"{avg_bpp:.6f}", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_psnr_rgb:.4f}", f"{avg_ssim_rgb:.4f}"])

    print(f"完成! 结果保存在 {out_csv}")

if __name__ == "__main__":
    run_test()