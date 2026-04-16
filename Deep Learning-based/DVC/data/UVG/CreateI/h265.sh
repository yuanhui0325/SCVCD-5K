##!/bin/bash
#rm -rf out
#rm result.txt
#touch result.txt
#
#echo "crf $1 resolution $2 x $3"
#
#for input in ../videos_crop/*.yuv; do
#
#	echo "orginal"
#
#	echo "-------------------"
#
#	echo "$input"
#
#	mkdir -p out/source/
#
#	ffmpeg -pix_fmt yuv420p -s:v $2x$3 -i $input -f image2 out/source/img%06d.png
#
#	mkdir -p out/h265/
#
#	FFREPORT=file=ffreport.log:level=56 ffmpeg -pix_fmt yuv420p -s $2x$3 -i $input -c:v libx265 -tune zerolatency -x265-params "crf=$1:keyint=12:verbose=1" out/h265/out.mkv
#
#	ffmpeg -i out/h265/out.mkv -f image2 out/h265/img%06d.png
#
#    mkdir -p ${input%.*}/H265L$1/
#	ffmpeg -i out/h265/out.mkv -f image2 ${input%.*}/H265L$1/im%04d.png
#    echo $input
#
#	CUDA_VISIBLE_DEVICES=7 python3 measure265.py $input $2 $3  >> result.txt
#
#	rm -rf out/h265
#
#	rm ffreport.log
#
#	rm -rf out/source
#
#	echo "-------------------"
#
#done
#
#python3 report.py



#
##!/bin/bash
#
## ================= 配置区域 =================
#DATA_DIR="/data/lichaofei/data/SCVCD-NEW/test/sequences"
#OUT_ROOT="/data/lichaofei/data/SCVCD-NEW/test/sequences-DVC"
#GOP_SIZE=12
## ===========================================
#
#if [ -z "$1" ]; then
#    echo "❌ Usage: bash gen_vimeo_final.sh [CRF_VALUE]"
#    exit 1
#fi
#
#CRF=$1
#RESULT_FILE="$(pwd)/result_crf${CRF}.txt"
#
## 清空结果文件
#rm -f "$RESULT_FILE"
#touch "$RESULT_FILE"
#
#echo "=============================================="
#echo "🚀 Processing Vimeo-90k (Final Fix)"
#echo "   CRF: $CRF"
#echo "=============================================="
#
#for group_dir in "$DATA_DIR"/*; do
#    if [ -d "$group_dir" ]; then
#        group_name=$(basename "$group_dir")
#        
#        for seq_dir in "$group_dir"/*; do
#            if [ -d "$seq_dir" ]; then
#                seq_name=$(basename "$seq_dir")
#                full_name="${group_name}/${seq_name}"
#                
#                # 1. 找图 & 确定分辨率
#                first_img=$(find "$seq_dir" -name "*.png" | head -n 1)
#                
#                if [ -z "$first_img" ]; then
#                    continue
#                fi
#
#                w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 "$first_img")
#                h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 "$first_img")
#
#                target_w=$(( (w / 64) * 64 ))
#                target_h=$(( (h / 64) * 64 ))
#
#                # 2. 准备路径
#                out_dir="$OUT_ROOT/$group_name/$seq_name/H265L$CRF"
#                mkdir -p "$out_dir"
#
#                # === 核心修复：更简单的判断逻辑 ===
#                # 检查是否存在 im0001.png (4位补零)
#                if [ -f "$seq_dir/im0001.png" ]; then
#                    input_pattern="$seq_dir/im%04d.png"
#                else
#                    # 否则默认使用 im1.png (无补零)
#                    input_pattern="$seq_dir/im%d.png"
#                fi
#                
#                out_pattern="$out_dir/im%04d.png"
#
#                # 3. 计算第一帧 BPP
#                bin_file_temp="temp_first_frame.265"
#                
#                # 增加 -start_number 1 以防万一
#                ffmpeg -y -hide_banner -loglevel error \
#                    -f image2 -start_number 1 -i "$input_pattern" \
#                    -vf "crop=${target_w}:${target_h}:0:0" \
#                    -c:v libx265 -x265-params "crf=${CRF}:keyint=${GOP_SIZE}:verbose=1" \
#                    -frames:v 1 \
#                    "$bin_file_temp"
#
#                if [ ! -f "$bin_file_temp" ]; then
#                    echo "❌ Failed to create .265 file for $full_name"
#                    continue
#                fi
#
#                filesize=$(wc -c < "$bin_file_temp")
#                # Python计算 (修复语法)
#                bpp=$(python3 -c "print(($filesize * 8) / ($target_w * $target_h))")
#                
#                echo "$full_name $bpp" >> "$RESULT_FILE"
#                rm -f "$bin_file_temp"
#
#                # 4. 生成完整序列
#                video_temp="temp_seq.mkv"
#
#                ffmpeg -y -hide_banner -loglevel error \
#                    -f image2 -start_number 1 -i "$input_pattern" \
#                    -vf "crop=${target_w}:${target_h}:0:0" \
#                    -c:v libx265 -x265-params "crf=${CRF}:keyint=${GOP_SIZE}:verbose=1" \
#                    -pix_fmt yuv420p \
#                    "$video_temp"
#
#                ffmpeg -y -hide_banner -loglevel error \
#                    -i "$video_temp" \
#                    -f image2 \
#                    "$out_pattern"
#
#                rm -f "$video_temp"
#
#                echo "✅ $full_name | BPP: $bpp"
#            fi
#        done
#    fi
#done
#
#echo "🎉 Done! Results saved to: $RESULT_FILE"             

#!/bin/sh
set -eu

# ============== 配置区：按需修改 ==============
DATA_DIR="/data/lichaofei/data/SCVCD-NEW/test/sequences"
OUT_ROOT="/data/lichaofei/data/SCVCD-NEW/test/sequences-DVC"

# 你用 7 帧 clip 测试 DVC：建议 GOP_SIZE=7
# 若你想按 UVG-style：GOP_SIZE=12
GOP_SIZE=7

# 是否额外生成整段解码帧（一般DVC只需要I帧im0001.png，建议0）
SAVE_FULL_SEQ=0
# ==============================================

# x265 编码参数（不改 CRF 档位的前提下，尽量偏向客观指标）
X265_PRESET="veryslow"   # 可改：medium/slow/veryslow
X265_TUNE="psnr"         # 可改：psnr/ssim/或留空 ""

# 可选：你想试 no-sao=1，就把下面改成 ":no-sao=1"
X265_EXTRA_PARAMS=""

if [ $# -lt 1 ]; then
  echo "Usage: sh h265.sh CRF"
  exit 1
fi

CRF="$1"
RESULT_FILE="$(pwd)/result_crf${CRF}.txt"
: > "$RESULT_FILE"

echo "CRF=$CRF  GOP_SIZE=$GOP_SIZE"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "RESULT_FILE=$RESULT_FILE"
echo "PRESET=$X265_PRESET  TUNE=$X265_TUNE  EXTRA=$X265_EXTRA_PARAMS"
echo "SAVE_FULL_SEQ=$SAVE_FULL_SEQ"
echo "--------------------------------------------"

# 逐个 clip 处理：DATA_DIR/group/clip
for group_dir in "$DATA_DIR"/*; do
  [ -d "$group_dir" ] || continue
  group_name=$(basename "$group_dir")

  for clip_dir in "$group_dir"/*; do
    [ -d "$clip_dir" ] || continue
    clip_name=$(basename "$clip_dir")
    key="${group_name}/${clip_name}"

    # 找第一张图（支持 PNG/JPG）
    first_img=$(find "$clip_dir" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
      | sort | head -n 1 || true)
    [ -n "$first_img" ] || continue

    base=$(basename "$first_img")
    stem=${base%.*}                       # im0345
    ext=${base##*.}                       # png / PNG / jpg ...
    num=$(printf "%s" "$stem" | sed 's/[^0-9]//g')     # 0345
    prefix=$(printf "%s" "$stem" | sed 's/[0-9]*$//')  # im
    digits=${#num}

    # 无数字则跳过
    if [ "$digits" -eq 0 ]; then
      echo "Skip $key (cannot parse frame number from $base)"
      continue
    fi

    # 起始编号（避免前导0当八进制）
    start=$(python3 -c "print(int('$num'))")

    input_pattern="${clip_dir}/${prefix}%0${digits}d.${ext}"

    # 读分辨率并 crop 到 64 的整数倍（左上角crop）
    wh=$(python3 - <<PY
from PIL import Image
im = Image.open(r"$first_img")
print(im.size[0], im.size[1])
PY
)
    w=$(printf "%s" "$wh" | awk '{print $1}')
    h=$(printf "%s" "$wh" | awk '{print $2}')
    target_w=$(( (w / 64) * 64 ))
    target_h=$(( (h / 64) * 64 ))

    if [ "$target_w" -lt 64 ] || [ "$target_h" -lt 64 ]; then
      echo "Skip $key (too small after 64-align: ${w}x${h} -> ${target_w}x${target_h})"
      continue
    fi

    out_dir="${OUT_ROOT}/${group_name}/${clip_name}/H265L${CRF}"
    mkdir -p "$out_dir"
    i_png="${out_dir}/im0001.png"
    full_pattern="${out_dir}/im%04d.png"

    # ---------- 1) 用同一次码流：生成 I 帧图 + 统计 I 帧 bpp ----------
    tmp_hevc="$(mktemp /tmp/first_frame_XXXXXX.hevc)"

    # 编第一帧为 raw HEVC（强制 yuv420p），并固定 keyint
    if [ -n "$X265_TUNE" ]; then
      TUNE_ARGS="-tune $X265_TUNE"
    else
      TUNE_ARGS=""
    fi

    ffmpeg -y -hide_banner -loglevel error \
      -f image2 -start_number "$start" -i "$input_pattern" \
      -vf "crop=${target_w}:${target_h}:0:0,format=yuv420p" \
      -frames:v 1 \
      -c:v libx265 -pix_fmt yuv420p -preset "$X265_PRESET" $TUNE_ARGS \
      -x265-params "crf=${CRF}:keyint=${GOP_SIZE}:min-keyint=${GOP_SIZE}:scenecut=0:repeat-headers=1:verbose=0${X265_EXTRA_PARAMS}" \
      -f hevc "$tmp_hevc"

    # bpp（按这一帧码流大小 / (W*H)）
    filesize=$(wc -c < "$tmp_hevc")
    bpp=$(python3 -c "print(($filesize*8)/($target_w*$target_h))")
    printf "%s %s\n" "$key" "$bpp" >> "$RESULT_FILE"

    # 解码这条 hevc 码流得到 I 帧参考图（与 bpp 严格对应）
    ffmpeg -y -hide_banner -loglevel error \
      -i "$tmp_hevc" -frames:v 1 "$i_png"

    rm -f "$tmp_hevc"

    # ---------- 2) 可选：生成整段序列的解码帧 ----------
    if [ "$SAVE_FULL_SEQ" -eq 1 ]; then
      tmp_mkv="$(mktemp /tmp/seq_XXXXXX.mkv)"
      ffmpeg -y -hide_banner -loglevel error \
        -f image2 -start_number "$start" -i "$input_pattern" \
        -vf "crop=${target_w}:${target_h}:0:0,format=yuv420p" \
        -c:v libx265 -pix_fmt yuv420p -preset "$X265_PRESET" $TUNE_ARGS \
        -x265-params "crf=${CRF}:keyint=${GOP_SIZE}:min-keyint=${GOP_SIZE}:scenecut=0:repeat-headers=1:verbose=0${X265_EXTRA_PARAMS}" \
        "$tmp_mkv"

      ffmpeg -y -hide_banner -loglevel error \
        -i "$tmp_mkv" -f image2 "$full_pattern"

      rm -f "$tmp_mkv"
    fi

    echo "OK $key  ${w}x${h}->${target_w}x${target_h}  start=$start digits=$digits  I_bpp=$bpp"
  done
done

echo "--------------------------------------------"
echo "Done. Results saved to: $RESULT_FILE"