#!/bin/bash

# 替换为你要复制的原文件名
original_file_1="sample_result_public.json"
# 替换为你想要的新文件名
new_file_1="result_public.json"

# 新增第二个文件的原文件名和新文件名
original_file_2="sample_result_private.json"
new_file_2="result_private.json"

# 定义一个函数来复制文件
copy_file() {
    local original="$1"
    local new="$2"
    if [ -f "$original" ]; then
        cp "$original" "$new"
        echo "文件 $original 已成功复制为 $new"
    else
        echo "原文件 $original 不存在，请检查路径。"
    fi
}

# 调用函数复制第一个文件
copy_file "$original_file_1" "$new_file_1"
# 调用函数复制第二个文件
copy_file "$original_file_2" "$new_file_2"

# 使用相对路径执行 tests.py
python intentvc_inference.py --num_beams 3

# 将两个新文件压缩成一个 zip 文件
zip result "result_public.json" "result_private.json"


# 检查压缩是否成功
if [ $? -eq 0 ]; then
    echo "文件 $new_file_1 和 $new_file_2 已成功压缩为 result.zip"
else
    echo "压缩文件时出错，请检查文件是否存在。"
fi
