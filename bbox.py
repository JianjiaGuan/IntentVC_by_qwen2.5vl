"""
This code reads a video file and an accompanying text file containing bounding box coordinates.
It overlays the bounding boxes onto the corresponding video frames and saves a new annotated video.
The output video is saved with a dynamically adjusted FPS to shorten the total duration to around 15 seconds.

@ IntentVC Challenge, Date: 2025-03-12
"""

import cv2
import numpy as np
import os

# 定义输入和输出目录
input_base_folder = "./IntentVCDatasets/IntentVC"
output_base_folder = "./data/video/"

# 确保输出目录存在
os.makedirs(output_base_folder, exist_ok=True)

# 目标视频时长（秒）
target_duration = 10

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

            output_video_path = os.path.join(output_base_folder, f"{video_name}.mp4")

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
            new_fps = frame_count / target_duration if original_duration > target_duration else original_fps

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, new_fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Error creating video writer for: {output_video_path}")
                cap.release()
                continue

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # # Draw bounding boxes on the frame
                # if frame_idx < len(bboxes):
                #     x, y, w, h = bboxes[frame_idx]
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Write frame to output video
                out.write(frame)
                frame_idx += 1

            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print(f"Annotated video saved at: {output_video_path}")