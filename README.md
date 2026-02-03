<div align="center">
<!-- <h1><img src="public/avatar.png" width="60px">Mind-Brush</h1> -->
<img src="public/logo_light.png" width="180px" align="center">

<h2>Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation</h2>

<p align="center">
  <b>English</b> | <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.01756" target="_blank">
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


## 📰 News

- **[2026-02-02]** 🔥 We have released [Paper](https://arxiv.org/abs/2602.01756), [Code](https://github.com/PicoTrex/Mind-Brush) and [Dataset](https://huggingface.co/datasets/PicoTrex/Mind-Brush)!

## 🏆 Contributions

- 🧠 **Mind-Brush Framework**: A novel agentic paradigm that unifies **Intent Analysis**, **Multi-modal Search**, and **Knowledge Reasoning** into a seamless **"Think-Research-Create"** workflow for image generation.
- 📊 **Mind-Bench**: A specialized benchmark designed to evaluate generative models on **dynamic external knowledge** and **complex logical deduction**, exposing the reasoning gaps in current SOTA multimodal models.
- 🏆 **Superior Performance**: 
  -  Elevates Qwen-Image baseline accuracy from ***0.02*** to ***0.31*** on Mind-Bench.
  -  Outperforms existing baselines on **WISE** (***+25.8%*** WiScore) and **RISEBench** (***+27.3%*** Accuracy).

## 📽️ Demo

| Search Case | Search & Reason Case |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/36d97bad-e94f-4bc2-aae8-5d54451b93bc" controls></video> | <video src="https://github.com/user-attachments/assets/c0639fab-f1d0-4ffd-9120-d43e418f2bd3" controls></video> |



## 🚀 Quickstart

**1. Clone**

```bash
git clone https://github.com/PicoTrex/Mind-Brush.git
cd Mind-Brush
```

**2. Install**

```bash
conda create -n mindbrush python=3.12
conda activate mindbrush
pip install -r requirements.txt
```

**3. Configuration**

Fill in the `[required]` fields in `config.yaml` (e.g., API keys, path settings).

> [!NOTE]
> You can set your language in `.chainlit/config.toml`. In Default, the language is set to `en-US`. You can change it to Chinese by setting `language = "zh-CN"`.
> [We only support English and Chinese for now. If you want to support other languages, you can add the corresponding language settings in the `locales` folder.]

**4. Launch**

Run the application with the following command:

```bash
chainlit run app.py -w
```
Once started, access the dashboard at <http://localhost:8000>.

## 🩷 Acknowledgement

* [chainlit](https://github.com/Chainlit/chainlit)
* [PaperGallery](https://github.com/LongHZ140516/PaperGallery)
* [Awesome-Nano-Banana-images](https://github.com/PicoTrex/Awesome-Nano-Banana-images)
