#import os
#import glob
#import subprocess
#import re
#import csv
#import numpy as np
#import torch
#from pytorch_msssim import ms_ssim
#
## ==================== 配置区域 (请修改) ====================
#
## 1. VTM 编码器路径
#VTM_EXEC = "/home/chaofeili/Tradition/VVCSoftware_VTM/bin/EncoderAppStatic"
#
## 2. 配置文件 (SCC 推荐 encoder_lowdelay_vtm.cfg 或 encoder_intra_vtm.cfg)
#CONFIG_FILE = "/home/chaofeili/Tradition/VVCSoftware_VTM/cfg/encoder_randomaccess_vtm_scc.cfg"
#
## 3. YUV 数据集路径
#DATASET_DIR = "/data/lichaofei/data/SCVCD-NEW-tratest"
#
## 4. 结果保存路径
#OUTPUT_DIR = "/data/lichaofei/data/VVC-SCC_results"
#
## 5. QP 设置
#QP_LIST = [22, 27, 32, 37, 42]
#
## 6. YUV 格式设置
## 如果是 SCC，通常建议保留 444。如果您的源文件是 420，请填 "yuv420p"
## 注意：VTM 即使输入 8bit，输出通常也是 10bit (yuv420p10le)
#INPUT_PIX_FMT = "yuv444p"  # 源 YUV 格式 (yuv420p 或 yuv444p)
#INPUT_BITDEPTH = 8         # 源 YUV 位深
#
## ==========================================================
#
#def parse_filename(filepath):
#    """从文件名解析宽、高、帧率"""
#    filename = os.path.basename(filepath)
#    # 匹配 1920x1080 这种格式
#    res_match = re.search(r'(\d+)x(\d+)', filename)
#    fps_match = re.search(r'_(\d+)\.yuv', filename) # 匹配 _30.yuv
#    
#    if not res_match:
#        print(f"Error: 文件名 {filename} 未包含分辨率 (例如 1920x1080)")
#        return None, None, None
#        
#    width = int(res_match.group(1))
#    height = int(res_match.group(2))
#    fps = int(fps_match.group(1)) if fps_match else 30 # 默认30fps
#    
#    return width, height, fps
#
#def calculate_msssim_and_psnr(width, height, frames, src_path, rec_path, device):
#    """
#    读取源YUV和重建YUV -> 转RGB -> 计算 MS-SSIM 和 PSNR
#    """
#    # 确定像素格式大小
#    # 420: 1.5 bytes/pixel, 444: 3 bytes/pixel
#    # 但我们这里通过 ffmpeg 转 RGB24 读取，所以固定是 H*W*3 字节
#    frame_size_bytes = width * height * 3
#
#    # 构造 FFmpeg 命令：读取原始 YUV (8bit) -> RGB Pipe
#    cmd_src = [
#        'ffmpeg', '-v', 'error',
#        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', 
#        '-pix_fmt', INPUT_PIX_FMT, '-i', src_path,
#        '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-'
#    ]
#
#    # 构造 FFmpeg 命令：读取重建 YUV (VTM输出通常是10bit) -> RGB Pipe
#    # VTM output usually matches input chroma but is 10-bit internal
#    rec_fmt = INPUT_PIX_FMT + '10le' if '10' not in INPUT_PIX_FMT else INPUT_PIX_FMT
#    cmd_rec = [
#        'ffmpeg', '-v', 'error',
#        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', 
#        '-pix_fmt', rec_fmt, '-i', rec_path,
#        '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-'
#    ]
#
#    p_src = subprocess.Popen(cmd_src, stdout=subprocess.PIPE)
#    p_rec = subprocess.Popen(cmd_rec, stdout=subprocess.PIPE)
#
#    total_msssim = 0.0
#    total_psnr = 0.0
#    valid_frames = 0
#
#    try:
#        for _ in range(frames):
#            # 从管道读取一帧 RGB 数据
#            raw_src = p_src.stdout.read(frame_size_bytes)
#            raw_rec = p_rec.stdout.read(frame_size_bytes)
#
#            if not raw_src or not raw_rec: break
#
#            # 转为 Tensor [1, 3, H, W], 归一化 0-1
#            src_np = np.frombuffer(raw_src, dtype=np.uint8).reshape((height, width, 3))
#            rec_np = np.frombuffer(raw_rec, dtype=np.uint8).reshape((height, width, 3))
#
#            # 计算 PSNR (在 RGB 域)
#            mse = np.mean((src_np - rec_np) ** 2)
#            if mse == 0: psnr = 100
#            else: psnr = 20 * np.log10(255.0 / np.sqrt(mse))
#            total_psnr += psnr
#
#            # 准备 MS-SSIM Tensor
#            src_t = torch.from_numpy(src_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
#            rec_t = torch.from_numpy(rec_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
#
#            # 计算 MS-SSIM
#            # win_size 默认11, 如果图太小会报错，一般 1920x1080 没问题
#            val = ms_ssim(src_t, rec_t, data_range=1.0, size_average=True)
#            total_msssim += val.item()
#            
#            valid_frames += 1
#
#    except Exception as e:
#        print(f"Error computing metrics: {e}")
#    finally:
#        p_src.stdout.close()
#        p_rec.stdout.close()
#        p_src.terminate()
#        p_rec.terminate()
#
#    if valid_frames == 0: return 0, 0
#    return total_msssim / valid_frames, total_psnr / valid_frames
#
#def run_test():
#    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
#    
#    # 结果文件
#    csv_file = open(os.path.join(OUTPUT_DIR, 'results_vvc.csv'), 'w', newline='')
#    writer = csv.writer(csv_file)
#    writer.writerow(['Sequence', 'QP', 'Bitrate(kbps)', 'BPP', 'PSNR_Y_Log', 'PSNR_RGB_Calc', 'MS-SSIM', 'Width', 'Height', 'Time(s)'])
#
#    # 查找所有 YUV
#    yuv_files = glob.glob(os.path.join(DATASET_DIR, "*.yuv"))
#    print(f"找到 {len(yuv_files)} 个 YUV 文件...")
#    
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"使用设备计算 MS-SSIM: {device}")
#
#    for yuv_path in yuv_files:
#        seq_name = os.path.basename(yuv_path)
#        
#        # 1. 解析文件名获取分辨率
#        width, height, fps = parse_filename(yuv_path)
#        if not width: continue # 跳过无法解析的文件
#        
#        # 2. 计算总帧数 (根据文件大小倒推)
#        file_size = os.path.getsize(yuv_path)
#        
#        # 计算每帧字节数
#        if "444" in INPUT_PIX_FMT:
#            bytes_per_frame = width * height * 3
#        else: # 420
#            bytes_per_frame = width * height * 1.5
#            
#        if INPUT_BITDEPTH == 10: bytes_per_frame *= 2 # 10bit 占2字节
#            
#        frame_count = int(file_size // bytes_per_frame)
#        print(f"\n序列: {seq_name} | {width}x{height} | {frame_count} 帧")
#
#        for qp in QP_LIST:
##            bin_file = os.path.join(OUTPUT_DIR, "temp.bin")
##            rec_yuv = os.path.join(OUTPUT_DIR, "rec.yuv")
#            bin_file = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}.bin")
#            rec_yuv  = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}_rec.yuv")
#            
#            # 3. 运行 VVC 编码
#            cmd = [
#                VTM_EXEC,
#                '-c', CONFIG_FILE,
#                '-i', yuv_path,
#                '-b', bin_file,
#                '-o', rec_yuv,
#                '-wdt', str(width),
#                '-hgt', str(height),
#                '-fr', str(fps),
#                '-f', str(frame_count),
#                '-q', str(qp),
#                f'--InputBitDepth={INPUT_BITDEPTH}',
#                '--InternalBitDepth=10', # VVC 内部处理通常为 10bit
#                '--ConformanceWindowMode=1', # 自动裁剪填充
#                # === 关键：显式开启 SCC 工具 (覆盖默认配置) ===
#                '--IBC=1',              # 开启帧内块拷贝
#                '--PLT=1',      # 开启调色板模式
#                '--HashME=1',           # 开启哈希运动估计 (非常重要!)
#                '--BDPCM=1',            # 开启 BDPCM
#                '--TransformSkip=1',    # 开启变换跳过
#                '--TransformSkipFast=1',# 加速变换跳过
#                
#                # === 针对 RA 的设置 ===
#                # RA 不需要设置 IntraPeriod=-1，它默认是 32 或 64，适合 300 帧视频
#                # 只有当你不想有 I 帧刷新（比如纯静态画面测试）才改 IntraPeriod
#           
#            ]
#           
#
#            # 处理色度格式
#            if "444" in INPUT_PIX_FMT:
#                cmd.append('--InputChromaFormat=444')
#            else:
#                cmd.append('--InputChromaFormat=420')
#
#            # print(f"  Encoding QP {qp}...")
##            ret = subprocess.run(cmd, capture_output=True, text=True)
##            log = ret.stdout
#            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#            if ret.returncode != 0:
#                print(f"Error: 编码失败 QP {qp}, returncode={ret.returncode}")
#                print("CMD:", " ".join(cmd))
#                print("STDERR:", ret.stderr[:1200])
#                continue
#            log = ret.stdout + "\n" + ret.stderr
#            
#            
#            # 4. 从 Log 获取 Bitrate 和 VTM 自带的 Y-PSNR
#            bitrate_log = 0.0
#            psnr_y_log = 0.0
#            enc_time = 0.0
#            
#            for line in log.splitlines():
#                if "Total Time" in line:
#                    try: enc_time = float(line.split()[2])
#                    except: pass
#                if "     a     " in line: # VTM 汇总行
#                    try:
#                        parts = line.split()
#                        bitrate_log = float(parts[2])
#                        psnr_y_log = float(parts[3])
#                    except: pass
#
#            # 5. 计算 BPP (Bits Per Pixel)
#            if os.path.exists(bin_file):
#                total_bits = os.path.getsize(bin_file) * 8
#                total_pixels = width * height * frame_count
#                bpp = total_bits / total_pixels
#            else:
#                print(f"Error: 编码失败 QP {qp}")
#                continue
#
#            # 6. 计算 MS-SSIM (使用 PyTorch)
#            # print(f"  Calculating MS-SSIM QP {qp}...")
#            avg_msssim, avg_psnr_rgb = calculate_msssim_and_psnr(
#                width, height, frame_count, yuv_path, rec_yuv, device
#            )
#            
#            print(f"  QP{qp}: BPP={bpp:.4f}, PSNR(Y)={psnr_y_log:.2f}, MS-SSIM={avg_msssim:.4f}")
#            writer.writerow([seq_name, qp, bitrate_log, bpp, psnr_y_log, avg_psnr_rgb, avg_msssim, width, height, enc_time])
#
#            # 清理
#            if os.path.exists(bin_file): os.remove(bin_file)
#            if os.path.exists(rec_yuv): os.remove(rec_yuv)
#
#    csv_file.close()
#    print("测试完成。结果保存在 Results_VTM_SCC 目录。")
#
#if __name__ == "__main__":
#    run_test()


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

VTM_EXEC = "/home/chaofeili/Tradition/VVCSoftware_VTM/bin/EncoderAppStatic"

# 如果这个 cfg 不存在或不适配，改成 encoder_randomaccess_vtm.cfg（你手动跑通的那份）
CONFIG_FILE = "/home/chaofeili/Tradition/VVCSoftware_VTM/cfg/encoder_randomaccess_vtm_scc.cfg"
# CONFIG_FILE = "/home/chaofeili/Tradition/VVCSoftware_VTM/cfg/encoder_randomaccess_vtm.cfg"

DATASET_DIR = "/data/lichaofei/data/SCVCD-NEW-tratest"
OUTPUT_DIR  = "/data/lichaofei/data/VVC-SCC_results"

QP_LIST = [22, 27, 32, 37, 42]

INPUT_PIX_FMT = "yuv444p"   # yuv444p 或 yuv420p
INPUT_BITDEPTH = 8          # 源 yuv 位深：8 或 10

# VTM InternalBitDepth=10 时，Recon 往往以 10bit(16bit little-endian) 写盘
REC_BITDEPTH = 10

FPS_DEFAULT = 30

# 是否同时计算 RGB-PSNR / RGB-MS-SSIM（非常慢）
DO_RGB = False

# ===================================================

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    res_match = re.search(r'(\d+)x(\d+)', filename)
    fps_match = re.search(r'_(\d+)\.yuv$', filename)  # 例如 _30.yuv；没有就默认30
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
    else:  # 420
        return int(w * h * 3 / 2) * bps

def read_y_plane(fp, w, h, pix_fmt, bitdepth):
    """读取一帧的Y平面，并跳过UV"""
    bps = 2 if bitdepth > 8 else 1
    y_bytes = w * h * bps
    y_buf = fp.read(y_bytes)
    if len(y_buf) != y_bytes:
        return None

    if bitdepth > 8:
        y = np.frombuffer(y_buf, dtype=np.uint16).reshape(h, w)
    else:
        y = np.frombuffer(y_buf, dtype=np.uint8).reshape(h, w)

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
      - avg_y_psnr
      - avg_y_msssim
      - sse_sum (sum of squared error over ALL pixels of ALL frames, after bitdepth alignment)
      - frames_cnt
      - pixels_cnt (w*h*frames_cnt)
    注意：
      如果 rec_bitdepth > src_bitdepth，会右移到 src_bitdepth 并转 uint8，以便 torch 兼容。
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

            # 对齐 bitdepth：10->8 右移2位，并转 uint8（避免 torch 不支持 uint16）
            if shift > 0:
                y1 = (y1 >> shift).astype(np.uint8)
            else:
                if y1.dtype != np.uint8:
                    y1 = y1.astype(np.uint8)

            # y0 也确保 uint8（src_bitdepth=8 时应当就是 uint8）
            if y0.dtype != np.uint8:
                y0 = y0.astype(np.uint8)

            # SSE / MSE / PSNR(Y)
            diff = y0.astype(np.float32) - y1.astype(np.float32)
            sse = float(np.sum(diff * diff))
            mse = sse / pixels_per_frame
            sse_sum += sse

            if mse < 1e-12:
                psnr = 100.0
            else:
                psnr = 10.0 * math.log10((max_src * max_src) / mse)
            sum_psnr += psnr

            # MS-SSIM(Y)
            t0 = torch.from_numpy(y0.copy()).unsqueeze(0).unsqueeze(0).float().to(device) / max_src
            t1 = torch.from_numpy(y1.copy()).unsqueeze(0).unsqueeze(0).float().to(device) / max_src
            sum_msssim += float(ms_ssim(t0, t1, data_range=1.0, size_average=True).item())

            cnt += 1

    if cnt == 0:
        return 0.0, 0.0, 0.0, 0, 0

    avg_psnr = sum_psnr / cnt
    avg_msssim = sum_msssim / cnt
    pixels_cnt = pixels_per_frame * cnt
    return avg_psnr, avg_msssim, sse_sum, cnt, pixels_cnt

@torch.no_grad()
def calc_rgb_metrics_ffmpeg(w, h, nframes, src_yuv, rec_yuv, device, input_pix_fmt, src_bitdepth, rec_bitdepth):
    """
    通过 ffmpeg 把 YUV 转成 RGB24，再算 RGB-PSNR & RGB-MS-SSIM（很慢）
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

    per_seq_csv = os.path.join(OUTPUT_DIR, "results_vvc_scc_per_video.csv")
    dataset_csv = os.path.join(OUTPUT_DIR, "results_vvc_scc_dataset.csv")

    yuv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.yuv")))
    print(f"找到 {len(yuv_files)} 个 YUV 文件...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备计算 MS-SSIM: {device}")
    print(f"DO_RGB={DO_RGB} (RGB指标会非常慢)")

    # dataset 汇总（按 QP 汇总）
    # 对 Y-PSNR：用 SSE/Pixels 得到全局 MSE -> PSNR（更标准）
    dataset_stat = {
        qp: {
            "bits": 0,
            "pixels": 0,
            "sse": 0.0,
            "msssim_sum": 0.0,
            "frames": 0,
            # RGB 可选
            "rgb_psnr_sum": 0.0,
            "rgb_msssim_sum": 0.0,
            "rgb_frames": 0,
        } for qp in QP_LIST
    }

    with open(per_seq_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Sequence", "QP",
            "BPP",
            "PSNR_Y_VTMlog",
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
                bin_file = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}.bin")
                rec_yuv  = os.path.join(OUTPUT_DIR, f"{seq_name}_QP{qp}_rec.yuv")

                cmd = [
                    VTM_EXEC,
                    "-c", CONFIG_FILE,
                    "-i", yuv_path,
                    "-b", bin_file,
                    "-o", rec_yuv,
                    "-wdt", str(w),
                    "-hgt", str(h),
                    "-fr", str(fps),
                    "-f", str(frame_count),
                    "-q", str(qp),
                    f"--InputBitDepth={INPUT_BITDEPTH}",
                    "--InternalBitDepth=10",
                    # 如果报 Unknown option ConformanceWindowMode，就删掉下一行
                    "--ConformanceWindowMode=1",
                    "--InputChromaFormat=444" if "444" in INPUT_PIX_FMT else "--InputChromaFormat=420",

                    # SCC tools for VTM 23.13
                    "--IBC=1",
                    "--PLT=1",
                    "--HashME=1",
                    "--BDPCM=1",
                    "--TransformSkip=1",
                    "--TransformSkipFast=1",
                ]

                ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if ret.returncode != 0:
                    print(f"Error: 编码失败 QP {qp}, returncode={ret.returncode}")
                    print("CMD:", " ".join(cmd))
                    print("STDERR:", ret.stderr[:1200])
                    continue

                log = ret.stdout + "\n" + ret.stderr

                if not os.path.exists(bin_file) or not os.path.exists(rec_yuv):
                    print(f"Error: output not found QP {qp}")
                    continue

                # bpp
                bits = os.path.getsize(bin_file) * 8
                bpp = bits / (w * h * frame_count)

                # parse VTM log PSNR(Y) if possible
                psnr_y_vtm = 0.0
                for line in log.splitlines():
                    if "     a     " in line:
                        try:
                            parts = line.split()
                            psnr_y_vtm = float(parts[3])
                        except:
                            pass

                # calc Y metrics + SSE
                y_psnr, y_msssim, sse_sum, y_frames, y_pixels = calc_y_metrics_with_sse(
                    w, h, frame_count, yuv_path, rec_yuv,
                    INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH, device
                )

                # RGB (optional)
                if DO_RGB:
                    rgb_psnr, rgb_msssim, rgb_frames = calc_rgb_metrics_ffmpeg(
                        w, h, frame_count, yuv_path, rec_yuv,
                        device, INPUT_PIX_FMT, INPUT_BITDEPTH, REC_BITDEPTH
                    )
                    print(f"  QP{qp}: bpp={bpp:.4f}, Y-PSNR={y_psnr:.3f}, Y-MS-SSIM={y_msssim:.4f}, RGB-PSNR={rgb_psnr:.3f}, RGB-MS-SSIM={rgb_msssim:.4f}")
                else:
                    rgb_psnr, rgb_msssim, rgb_frames = "", "", 0
                    print(f"  QP{qp}: bpp={bpp:.4f}, Y-PSNR={y_psnr:.3f}, Y-MS-SSIM={y_msssim:.4f}")

                # 写入每个视频的结果
                writer.writerow([
                    seq_name, qp,
                    f"{bpp:.6f}",
                    f"{psnr_y_vtm:.4f}",
                    f"{y_psnr:.4f}", f"{y_msssim:.6f}",
                    f"{rgb_psnr}" if rgb_psnr != "" else "",
                    f"{rgb_msssim}" if rgb_msssim != "" else "",
                    w, h, y_frames
                ])

                # 汇总到整个数据集（按 QP）
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

                # 如需节省空间可删掉中间文件；调试期建议先保留
                # os.remove(bin_file)
                # os.remove(rec_yuv)

    # 输出数据集汇总（每个 QP 一行）
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

            # global MSE from SSE/pixels
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
    print("  per-video results :", per_seq_csv)
    print("  dataset summary   :", dataset_csv)

if __name__ == "__main__":
    run_test()