<div align="center">
<!-- <h1><img src="public/avatar.png" width="60px">Mind-Brush</h1> -->
<img src="public/logo_light.png" width="180px" align="center">

<h2>Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation</h2>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

<p align="center">
  <a href="" target="_blank">
    <img src="https://img.shields.io/badge/-arXiv-%23b91c1c?style=flat&logo=arxiv&logoColor=white&labelColor=%23b91c1c" alt="arXiv Paper">
  </a>
  <a href="https://github.com/PicoTrex/Mind-Brush" target="_blank">
    <img src="https://img.shields.io/badge/-Github-%234aa2a9?style=flat&logo=github&logoColor=white&labelColor=%234aa2a9" alt="GitHub Repo">
  </a>
  <a href="https://huggingface.co/datasets/PicoTrex/Mind-Brush" target="_blank">
    <img src="https://img.shields.io/badge/-Dataset-%23FFD21E?style=flat&logo=huggingface&logoColor=white&labelColor=%23FFD21E" alt="Dataset">
  </a>
</p>

<p align="center">
  <img src="assets/flag.jpg" width="90%">
</p>

</div>


## 📰 新闻

- **[2026-02-01]** 🔥 我们发布了 [论文](./)、[代码](https://github.com/PicoTrex/Mind-Brush) 和 [数据集](https://huggingface.co/datasets/PicoTrex/Mind-Brush)！

## 🏆 主要贡献

- 🧠 **Mind-Brush 框架**：一种全新的代理范式，将 **意图分析**、**多模态搜索** 和 **知识推理** 统一为一个无缝的 **“思考-研究-创作”** 图像生成工作流。
- 📊 **Mind-Bench**：专门设计的基准测试，用于评估生成模型在**动态外部知识**和**复杂逻辑推理**方面的表现，揭示了当前 SOTA 多模态模型的推理差距。
- 🏆 **卓越性能**：
  - **15倍提升**：在 Mind-Bench 上将 Qwen-Image 基准准确率从 ***0.02*** 提升至 ***0.31***。
  - **全新 SOTA**：在 **WISE**（***+25.8%*** WiScore）和 **RISEBench**（***+27.3%*** 准确率）上超越了现有基准。

## 📽️ 演示回放

| 案例 1 | 案例 2 |
| :---: | :---: |
| <video src="assets/case_1.mp4" width="400" controls></video> | <video src="assets/case_2.mp4" width="400" controls></video> |

## 🚀 快速开始

**1. 克隆项目**

```bash
git clone https://github.com/PicoTrex/Mind-Brush.git
cd Mind-Brush
```

**2. 安装环境**

```bash
conda create -n mindbrush python=3.12
conda activate mindbrush
pip install -r requirements.txt
```

**3. 配置**

填写 `config.yaml` 中的 `[required]` 字段（如 API 密钥、路径设置等）。

> [!NOTE]
> 你可以在 `.chainlit/config.toml` 中设置你的语言。默认情况下，语言设置为 `en-US`。你可以通过设置 `language = "zh-CN"` 将其更改为中文。
> [目前我们只支持英语和中文。如果你想支持其他语言，可以在 `locales` 文件夹中添加相应的语言设置。]

**4. 启动运行**

使用以下命令运行程序：

```bash
chainlit run app.py -w
```

启动后，通过浏览器访问仪表板：<http://localhost:8000>。

## 🩷 鸣谢

* [chainlit](https://github.com/Chainlit/chainlit)
* [PaperGallery](https://github.com/LongHZ140516/PaperGallery)
* [Awesome-Nano-Banana-images](https://github.com/PicoTrex/Awesome-Nano-Banana-images)
