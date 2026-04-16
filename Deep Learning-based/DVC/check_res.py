import os
from PIL import Image

# 你的数据集路径
base_path = "/data/lichaofei/data/SCVCD-NEW/test/sequences/"

print(f"{'Video Path':<35} | {'Resolution':<15} | {'Check (Mod 64)'}")
print("-" * 75)

# 第一层循环：遍历 00001, 00002...
for group_name in sorted(os.listdir(base_path)):
    group_path = os.path.join(base_path, group_name)
    
    if not os.path.isdir(group_path):
        continue

    # 第二层循环：遍历 0001, 0002... (这才是真正包含图片的文件夹)
    sub_folders = sorted(os.listdir(group_path))
    
    # 为了避免输出太长，每个大组我们只检查前2个子视频
    for clip_name in sub_folders[:2]: 
        clip_path = os.path.join(group_path, clip_name)
        
        if os.path.isdir(clip_path):
            # 找图片
            files = [f for f in os.listdir(clip_path) if f.endswith('.png')]
            
            if len(files) > 0:
                img_path = os.path.join(clip_path, files[0])
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        
                        # 检查是否能被 64 整除 (DVC 推荐) 或者 16 整除 (最低要求)
                        msg = "OK"
                        if w % 64 != 0 or h % 64 != 0:
                            msg = "⚠️ Need Crop!"
                        if w % 16 != 0 or h % 16 != 0:
                            msg = "❌ Error (Not mod 16)"
                            
                        print(f"{group_name}/{clip_name:<20} | {w}x{h:<9} | {msg}")
                except:
                    print(f"{group_name}/{clip_name:<20} | Error reading")
            else:
                print(f"{group_name}/{clip_name:<20} | No PNGs")