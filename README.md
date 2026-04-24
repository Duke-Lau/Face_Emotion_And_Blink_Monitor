# Face Emotion And Blink Monitor

这是一个从零重建的摄像头人脸分析项目：

- 通过摄像头实时检测人脸
- 对当前人脸进行情绪识别
- 对当前人脸进行眨眼识别并计数

主程序会打开摄像头窗口，实时显示人脸框、情绪标签、情绪置信度、眨眼次数、眼睛开合指标，以及 `py-feat` 提供的 Action Units 简报。

## 项目结构

```text
Project/
├── main.py
├── requirements.txt
├── README.md
├── src/face_monitor/
│   ├── app.py
│   ├── blink.py
│   ├── emotion.py
│   ├── models.py
│   └── tracking.py
└── tests/test_blink.py
```

## 技术方案

### 1. 人脸与关键点

使用 `MediaPipe Face Mesh` 从摄像头画面中实时提取人脸关键点。这样可以同时拿到：

- 人脸边界框
- 眼部关键点
- 嘴部和眉眼几何关系

这为情绪识别和眨眼识别提供同一套底层输入。

### 2. 眨眼识别

使用 `Eye Aspect Ratio, EAR` 算法判断眼睛开合程度：

- EAR 低于阈值，认为眼睛闭合
- 连续若干帧闭合，再恢复张开，计为一次眨眼

这样可以避免因为轻微抖动或单帧噪声导致误判。

### 3. 情绪识别与 Action Units

情绪识别主路径使用 `py-feat`：

- 使用 `retinaface + mobilefacenet + resmasknet + xgb` 组合做实时表情分析
- 输出离散情绪概率
- 输出 Action Units，便于后续做人因实验解释

如果当前机器上 `py-feat` 初始化失败，程序会自动退回到几何兜底情绪识别，保证流程仍然可运行。

### 4. 实验日志

程序每次启动都会生成一个 CSV 日志文件，记录：

- 时间戳
- 帧号
- face_id
- 眨眼计数
- EAR
- 人脸框
- 情绪标签与情绪概率
- Action Unit 强度

这类日志适合和实验刺激呈现时刻做对齐，再按 trial 或时间窗统计。

## 安装

推荐先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

`py-feat` 第一次初始化时可能会下载预训练模型，网络较慢时需要多等一会。

## 运行

最简单的启动方式：

```bash
python3 main.py --mirror --show-fps
```

指定摄像头和最多追踪人数：

```bash
python3 main.py --camera 0 --max-faces 2 --mirror --show-fps
```

强制使用 py-feat 情绪模型：

```bash
python3 main.py --emotion-backend py-feat
```

强制使用几何兜底情绪识别：

```bash
python3 main.py --emotion-backend geometry
```

## 运行时快捷键

- `q`：退出程序
- `r`：重置当前会话中的眨眼计数

## 关键参数

- `--emotion-interval`
  控制每张脸隔多少帧重新做一次情绪识别，默认 `6`

- `--blink-threshold`
  控制 EAR 阈值，默认 `0.23`

- `--min-closed-frames`
  连续闭眼多少帧算一次有效眨眼，默认 `2`

- `--emotion-log`
  指定 CSV 日志输出路径；不传时默认写到 `logs/`

## 当前版本边界

- 这是实时产品原型，重点放在“摄像头实时情绪识别 + 眨眼识别”两个主功能
- 目前没有做人脸身份注册和身份识别数据库
- `py-feat` 首次运行可能需要额外时间下载模型
- 几何兜底情绪识别只能作为降级方案，不能替代正式情绪模型

## 测试

项目包含一个轻量单元测试，覆盖：

- EAR 计算逻辑
- 眨眼状态机的计数逻辑

运行方式：

```bash
python3 -m pytest tests/test_blink.py
```
