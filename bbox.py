"""
This code reads a video file and an accompanying text file containing bounding box coordinates.
It performs two processing tasks:
1. Overlays the bounding boxes onto the corresponding video frames and saves with 10s duration
2. Expands the bounding boxes by 10 pixels, turns all information outside the expanded boxes black, 
   and saves with 30s duration

@ IntentVC Challenge, Date: 2025-03-12
"""

import cv2
import numpy as np
import os

# 定义输入和输出目录
input_base_folder = "./IntentVCDatasets/IntentVC"
output_base_folder_1 = "./data/video/"  # 第一个任务的输出目录
output_base_folder_2 = "./data/video_small/"  # 第二个任务的输出目录

# 确保输出目录存在
os.makedirs(output_base_folder_1, exist_ok=True)
os.makedirs(output_base_folder_2, exist_ok=True)

# 目标视频时长（秒）
target_duration_1 = 10  # 第一个任务的目标时长
target_duration_2 = 30  # 第二个任务的目标时长

# 遍历输入目录下的所有子目录
for root, dirs, files in os.walk(input_base_folder):
    for file in files:
        if file.endswith('.mp4'):
            video_name = os.path.splitext(file)[0]
            video_path = os.path.join(root, file)
            # 修改边界框文件路径，统一为 object_bboxes.txt
            bboxes_path = os.path.join(root, "object_bboxes.txt")

            # 检查输入文件是否存在
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue
            if not os.path.exists(bboxes_path):
                print(f"Bounding box file not found: {bboxes_path}")
                continue

            output_video_path_1 = os.path.join(output_base_folder_1, f"{video_name}.mp4")
            output_video_path_2 = os.path.join(output_base_folder_2, f"{video_name}.mp4")

            try:
                # Read bounding boxes
                with open(bboxes_path, 'r') as f:
                    bboxes = [list(map(int, line.strip().split(','))) for line in f]
            except Exception as e:
                print(f"Error reading bounding box file {bboxes_path}: {e}")
                continue

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)

            # 计算原视频时长（秒）
            original_duration = frame_count / original_fps

            # 动态计算新的帧率
            new_fps_1 = frame_count / target_duration_1 if original_duration > target_duration_1 else original_fps
            new_fps_2 = frame_count / target_duration_2 if original_duration > target_duration_2 else original_fps

            # 任务1：绘制边界框并保存
            print(f"Processing task 1 for {video_name}...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_1 = cv2.VideoWriter(output_video_path_1, fourcc, new_fps_1, (frame_width, frame_height))
            if not out_1.isOpened():
                print(f"Error creating video writer for: {output_video_path_1}")
                cap.release()
                continue

            # 重置视频到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw bounding boxes on the frame
                if frame_idx < len(bboxes):
                    x, y, w, h = bboxes[frame_idx]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Write frame to output video
                out_1.write(frame)
                frame_idx += 1

            # Release resources for task 1
            out_1.release()
            print(f"Task 1 completed - Annotated video saved at: {output_video_path_1}")

            # 任务2：扩展边界框并处理框外区域
            print(f"Processing task 2 for {video_name}...")
            out_2 = cv2.VideoWriter(output_video_path_2, fourcc, new_fps_2, (frame_width, frame_height))
            if not out_2.isOpened():
                print(f"Error creating video writer for: {output_video_path_2}")
                cap.release()
                continue

            # 重置视频到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < len(bboxes):
                    x, y, w, h = bboxes[frame_idx]
                    # 扩展边界框 10 个像素
                    x1 = max(0, x - 10)
                    y1 = max(0, y - 10)
                    x2 = min(frame_width, x + w + 10)
                    y2 = min(frame_height, y + h + 10)

                    # 将框外区域置为黑色
                    frame[:y1, :] = 0
                    frame[y2:, :] = 0
                    frame[:, :x1] = 0
                    frame[:, x2:] = 0

                # Write frame to output video
                out_2.write(frame)
                frame_idx += 1

            # Release resources for task 2
            out_2.release()
            print(f"Task 2 completed - Processed video saved at: {output_video_path_2}")

            # Release video capture
            cap.release()
            cv2.destroyAllWindows()

            print(f"Both tasks completed for {video_name}")
            print("-" * 50) 