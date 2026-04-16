#import os
#
## 配置你的路径
#ROOT_DIR = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
#OUT_FILE = "/data/lichaofei/data/SCVCD-NEW/train/dcvc-dc_train.txt"
#
#def regen_list():
#    lines = []
#    print(f"Scanning {ROOT_DIR} ...")
#    
#    # 遍历所有视频 (00001, 00002...)
#    for vid in sorted(os.listdir(ROOT_DIR)):
#        vid_path = os.path.join(ROOT_DIR, vid)
#        if not os.path.isdir(vid_path): continue
#        
#        # 遍历所有片段 (0001, 0002...)
#        for clip in sorted(os.listdir(vid_path)):
#            clip_path = os.path.join(vid_path, clip)
#            if not os.path.isdir(clip_path): continue
#            
#            # 检查关键文件是否存在 (im1.png)
#            if os.path.exists(os.path.join(clip_path, "im1.png")):
#                # 写入格式: 00001/0001
#                lines.append(f"{vid}/{clip}")
#
#    # 保存
#    with open(OUT_FILE, "w") as f:
#        f.write("\n".join(lines))
#        
#    print(f"✅ 完成！共找到 {len(lines)} 个有效片段。")
#    print(f"已保存到: {OUT_FILE}")
#
#if __name__ == "__main__":
#    regen_list()


#都用
#import os
#
## ================= 配置 =================
## 你的原始列表文件
#INPUT_LIST = "/data/lichaofei/data/SCVCD-NEW/train/dcvc-dc_train.txt"
## 你的序列根目录
#ROOT_DIR = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
## 清洗后的输出文件
#OUTPUT_LIST = "/data/lichaofei/data/SCVCD-NEW/train/dcvc-dc_train_clean.txt"
## 需要检查的帧 (DCVC-DC 通常需要 7 帧)
#REQUIRED_FRAMES = ["im1.png", "im2.png", "im3.png", "im4.png", "im5.png", "im6.png", "im7.png"]
## =======================================
#
#def clean_list():
#    if not os.path.exists(INPUT_LIST):
#        print("❌ 找不到输入列表文件")
#        return
#
#    print(f"正在扫描列表: {INPUT_LIST} ...")
#    with open(INPUT_LIST, 'r') as f:
#        lines = [line.strip() for line in f if line.strip()]
#
#    valid_lines = []
#    bad_count = 0
#
#    for idx, seq_name in enumerate(lines):
#        seq_path = os.path.join(ROOT_DIR, seq_name)
#        
#        is_valid = True
#        # 1. 检查文件夹是否存在
#        if not os.path.isdir(seq_path):
#            is_valid = False
#        else:
#            # 2. 检查所有帧是否存在
#            for frame in REQUIRED_FRAMES:
#                if not os.path.exists(os.path.join(seq_path, frame)):
#                    is_valid = False
#                    break
#        
#        if is_valid:
#            valid_lines.append(seq_name)
#        else:
#            bad_count += 1
#            if bad_count < 5: # 只打印前几个错误
#                print(f"   ⚠️ 发现坏数据 (剔除): {seq_name}")
#
#    # 保存新列表
#    with open(OUTPUT_LIST, 'w') as f:
#        f.write('\n'.join(valid_lines))
#
#    print("-" * 30)
#    print(f"原始数量: {len(lines)}")
#    print(f"有效数量: {len(valid_lines)}")
#    print(f"剔除坏数据: {bad_count}")
#    print(f"✅ 新列表已生成: {OUTPUT_LIST}")
#
#if __name__ == "__main__":
#    clean_list()

#只用每个视频的前两个7帧
import os

# ================= 配置 =================
# 1. 你的序列根目录
ROOT_DIR = "/data/lichaofei/data/SCVCD-NEW/train/sequences"
# 2. 新列表的保存路径
OUT_FILE = "/data/lichaofei/data/SCVCD-NEW/train/dcvc-dc_train.txt"
# 3. 每个视频保留几个片段？
MAX_CLIPS = 2
# =======================================

def generate_limit_list():
    if not os.path.exists(ROOT_DIR):
        print("❌ 错误：找不到目录")
        return

    valid_lines = []
    video_count = 0
    total_clips = 0

    print(f"正在扫描 {ROOT_DIR} ...")
    
    # 遍历视频 ID (00001, 00002...)
    for vid in sorted(os.listdir(ROOT_DIR)):
        vid_path = os.path.join(ROOT_DIR, vid)
        if not os.path.isdir(vid_path): continue
        
        video_count += 1
        
        # 获取该视频下的所有片段 (0001, 0002, 0003, 0004...)
        clips = sorted(os.listdir(vid_path))
        
        # 【关键】只取前 MAX_CLIPS 个
        selected_clips = clips[:MAX_CLIPS]
        
        for clip in selected_clips:
            clip_path = os.path.join(vid_path, clip)
            # 简单检查是否存在 im1.png，确保是有效片段
            if os.path.isdir(clip_path) and os.path.exists(os.path.join(clip_path, "im1.png")):
                valid_lines.append(f"{vid}/{clip}")
                total_clips += 1

    # 保存
    with open(OUT_FILE, 'w') as f:
        f.write('\n'.join(valid_lines))
        
    print(f"✅ 生成完毕！")
    print(f"   视频总数: {video_count}")
    print(f"   保留片段: {total_clips}")
    print(f"   新列表路径: {OUT_FILE}")

if __name__ == "__main__":
    generate_limit_list()


