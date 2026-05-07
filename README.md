# 具身导航 Agent

基于 Habitat-sim 仿真环境的具身导航 Agent，支持文字输入，控制机器人导航到指定地点。

## 功能

- 在 HM3D 室内场景中进行导航
- 支持文字指令输入，如"请到沙发旁边"
- 支持多目标导航：沙发、床、椅子
- 使用 OWL-ViT 零样本目标检测识别目标
- 基于 FBE（Frontier-Based Exploration）自主探索场景
- 仅使用 RGB 图像和深度图等视觉信息，不使用特权信息
- 提供 Web 界面展示导航过程

## 技术栈

- 仿真器：Habitat-sim
- 场景：HM3D（00800-TEEsavR23oF）
- 目标检测：OWL-ViT（zero-shot）
- 后端：Flask
- 前端：HTML / JavaScript

## 运行方法

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 启动服务
```bash
cd /root
python app.py
```

3. 打开终端输出的地址，输入导航指令即可

## 项目结构

```
├── app.py                # Flask 后端
├── navigator_process.py  # 导航核心逻辑
├── static/
│   └── index.html        # 前端页面
└── requirements.txt
```
