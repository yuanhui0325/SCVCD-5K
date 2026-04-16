#import os
#
## ================= 配置 =================
## 你的原始 clip 列表文件路径
#SRC_TRAIN_LIST = "/data/lichaofei/data/SCVCD-NEW/train/sep_trainlist.txt"
## 假设你有验证集列表（如果没有，暂时用训练集代替跑通）
#SRC_VAL_LIST   = "/data/lichaofei/data/SCVCD-NEW/val/sep_vallist.txt" 
#SRC_TEST_LIST   = "/data/lichaofei/data/SCVCD-NEW/test/sep_testlist.txt"
#
## 输出 DVC 格式列表的保存位置 (建议存在 data 目录下)
#DST_TRAIN_TXT = "/data/lichaofei/data/SCVCD-NEW/train/train.txt"
#DST_VAL_TXT   = "/data/lichaofei/data/SCVCD-NEW/val/val.txt"
#DST_TEST_TXT   = "/data/lichaofei/data/SCVCD-NEW/test/test.txt"
## 每一个 Clip 要拆解成哪些帧？
## DVC 逻辑是当前帧 - 2 = 参考帧。
## 所以我们取 im3, im4, im5, im6, im7 (对应的参考帧是 im1...im5)
#FRAMES = [3, 4, 5, 6, 7]
## =======================================
#
#def convert_list(src_path, dst_path):
#    if not os.path.exists(src_path):
#        print(f"❌ 找不到文件: {src_path}")
#        return
#
#    with open(src_path, 'r') as f:
#        clips = [line.strip() for line in f if line.strip()]
#
#    lines = []
#    for clip in clips:
#        # clip 类似 "00001/0001"
#        for i in FRAMES:
#            # 生成 "00001/0001/im3.png"
#            rel_path = f"{clip}/im{i}.png"
#            lines.append(rel_path)
#
#    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#    with open(dst_path, 'w') as f:
#        f.write("\n".join(lines) + "\n")
#    
#    print(f"✅ 已生成: {dst_path} (共 {len(lines)} 个样本)")
#
#if __name__ == "__main__":
#    convert_list(SRC_TRAIN_LIST, DST_TRAIN_TXT)
#    convert_list(SRC_VAL_LIST, DST_VAL_TXT)
#    convert_list(SRC_TEST_LIST, DST_TEST_TXT)

import os

# ================= 配置 =================
# 你的训练集 sequences 根目录
ROOT_DIR = "/data/lichaofei/data/SCVCD-NEW/test/sequences"

# 输出文件路径
OUT_FILE = "/data/lichaofei/data/SCVCD-NEW/test/test_cleaned.txt"

# 需要包含的帧 (DVC 训练用 im3-im7)
TARGET_FRAMES = [3, 4, 5, 6, 7]
# =======================================

def generate_clean_list():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ 错误: 找不到目录 {ROOT_DIR}")
        return

    print(f"🔍 正在扫描 {ROOT_DIR} ...")
    valid_lines = []
    
    # 遍历视频 ID (00001...)
    video_dirs = sorted(os.listdir(ROOT_DIR))
    for vid in video_dirs:
        vid_path = os.path.join(ROOT_DIR, vid)
        if not os.path.isdir(vid_path): continue
        
        # 遍历片段 ID (0001...)
        clip_dirs = sorted(os.listdir(vid_path))
        for clip in clip_dirs:
            clip_path = os.path.join(vid_path, clip)
            if not os.path.isdir(clip_path): continue
            
            # 【重要检查】DVC 需要 im(t-2) 作为参考帧
            # 如果我们要用 im3，那么 im1 必须存在。
            # 如果 im1 都不存在，这个片段就完全不能用。
            if not os.path.exists(os.path.join(clip_path, "im1.png")):
                continue

            # 检查并添加目标帧
            for i in TARGET_FRAMES:
                img_name = f"im{i}.png"
                img_path = os.path.join(clip_path, img_name)
                
                # 只有文件真实存在时才写入
                if os.path.exists(img_path):
                    # 写入绝对路径
                    valid_lines.append(img_path)

    # 保存文件
    with open(OUT_FILE, 'w') as f:
        f.write('\n'.join(valid_lines))
        
    print(f"✅ 生成完毕！")
    print(f"   有效样本数: {len(valid_lines)}")
    print(f"   保存路径: {OUT_FILE}")

if __name__ == "__main__":
    generate_clean_list()