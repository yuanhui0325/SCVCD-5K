import os

# ================= 配置 =================
# 你的原始 clip 列表文件路径
SRC_TRAIN_LIST = "/data/lichaofei/data/SCVCD-NEW/train/sep_trainlist.txt"
# 假设你有验证集列表（如果没有，暂时用训练集代替跑通）
SRC_VAL_LIST   = "/data/lichaofei/data/SCVCD-NEW/val/sep_vallist.txt" 
SRC_TEST_LIST   = "/data/lichaofei/data/SCVCD-NEW/test/sep_testlist.txt"

# 输出 DVC 格式列表的保存位置 (建议存在 data 目录下)
DST_TRAIN_TXT = "/data/lichaofei/data/SCVCD-NEW/train/train.txt"
DST_VAL_TXT   = "/data/lichaofei/data/SCVCD-NEW/val/val.txt"
DST_TEST_TXT   = "/data/lichaofei/data/SCVCD-NEW/test/test.txt"
# 每一个 Clip 要拆解成哪些帧？
# DVC 逻辑是当前帧 - 2 = 参考帧。
# 所以我们取 im3, im4, im5, im6, im7 (对应的参考帧是 im1...im5)
FRAMES = [3, 4, 5, 6, 7]
# =======================================

def convert_list(src_path, dst_path):
    if not os.path.exists(src_path):
        print(f"❌ 找不到文件: {src_path}")
        return

    with open(src_path, 'r') as f:
        clips = [line.strip() for line in f if line.strip()]

    lines = []
    for clip in clips:
        # clip 类似 "00001/0001"
        for i in FRAMES:
            # 生成 "00001/0001/im3.png"
            rel_path = f"{clip}/im{i}.png"
            lines.append(rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"✅ 已生成: {dst_path} (共 {len(lines)} 个样本)")

if __name__ == "__main__":
    convert_list(SRC_TRAIN_LIST, DST_TRAIN_TXT)
    convert_list(SRC_VAL_LIST, DST_VAL_TXT)
    convert_list(SRC_TEST_LIST, DST_TEST_TXT)



