# 基于llama-factory的Qwen2.5-VL 视频理解模型微调指南

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型准备](#模型准备)
- [模型微调](#模型微调)
- [模型推理](#模型推理)

## 环境配置

### 1. 基础环境安装

按照官方文档安装基础环境。

### 2. 额外依赖安装

```bash
# 安装 Qwen VL 工具包
pip install qwen-vl-utils

# 安装 ModelScope
pip install modelscope
```

### 3. 环境文件说明

- `requirements.txt`: 官方环境依赖列表
- `requirements_me.txt`: 当前项目环境依赖列表

## 数据准备

### 1. 数据集下载

从官方网站下载数据集：[IntentVC Dataset](https://sites.google.com/view/intentvc/dataset)

### 2. 目录结构

将下载的数据按以下结构放置在项目根目录：

```
项目根目录/
├── IntentVCDataset/
│   ├── IntentVC/
│   │   ├── ...
│   ├── ...其他文件
```

### 3. 数据预处理

执行以下命令处理数据：

```bash
python bbox.py

```

执行后将在 `data/` 目录下生成 `video/` 和 `video——small/`文件夹，包含所有处理后的视频文件。

## 模型准备

### 1. 下载基础模型

下载 `Qwen2.5-VL-7B-Instruct` 模型。

### 2. 模型放置

将下载的模型放置在项目根目录的 `Qwen/` 目录下：

```
项目根目录/
├── Qwen/
│   ├── Qwen2.5-VL-32B-Instruct/
```

## 模型微调

### 1. 启动训练界面

```bash
llamafactory-cli webui
```

### 2. 配置训练参数

#### 基本参数设置

- **模型名称**：`Qwen2.5-VL-7B-Instruct`
- **模型路径**：`Qwen/Qwen2.5-VL-7B-Instruct`
- **数据集路径**：`data/`
- **数据集类型**：`mllm_videos_with_obj_class`
- **训练轮次**：`3`
- **截断长度**：`5000+`

### 3. 开始训练

点击界面上的"训练"按钮，等待训练完成。

### 4. 导出模型

1. 设置检查点路径为训练输出目录
2. 在 Export 面板中设置自定义导出路径
3. 点击导出按钮

## 模型推理

### 1. 修改推理配置

修改项目根目录下 `intentvc_inference.py` 中的模型路径。

### 2. 执行推理

```
sh test.sh
```

### 3. 获取结果

执行完成后将生成 `result.zip` 文件，包含推理结果。

## 注意事项

- 确保有足够的磁盘空间存储数据集和模型
- 训练过程需要较大的GPU显存
- 建议定期备份训练过程中的检查点
