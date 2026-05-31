# Build_DeepSeek_Step_by_Step

`Build_DeepSeek_Step_by_Step` 是一个从零拆解现代 LLM 关键模块的实现型项目。它不尝试复刻完整工业训练系统，而是把一条典型的 DeepSeek-like 技术路线拆成清晰可读的 notebook 和教学辅助代码，让读者可以一路追踪：

- 文本如何变成 token
- token 如何变成 embedding
- attention 如何工作
- GQA、MLA、MoE 这类现代结构到底在解决什么工程问题
- 训练如何从 pretraining 一路走到 SFT、reward model、PPO、GRPO
- 数据如何从采集、清洗、去重一路走到可训练语料

这个项目的核心目标不是堆术语，而是把“模块、公式、代码、图、工程含义”放到同一条理解链上。

## 当前覆盖范围

项目目前已经覆盖 5 条主线：

1. **基础数学与输入表示**
   - 向量、矩阵、softmax、mask
   - tokenizer、BPE、embedding

2. **Transformer 基础结构**
   - self-attention
   - multi-head attention
   - RoPE
   - RMSNorm、residual、FFN、SwiGLU
   - basic decoder-only block

3. **推理效率与容量扩展**
   - KV cache
   - MQA / GQA
   - MLA
   - MoE

4. **训练与对齐链路**
   - pretraining
   - continued training
   - SFT
   - reward model
   - RLHF
   - PPO / DPO / GRPO

5. **数据工程链路**
   - data collection / crawling
   - main-content extraction
   - cleaning / filtering
   - exact dedup / near-duplicate detection
   - quality bucketing

额外还补了一节：

6. **位置编码可视化**
   - learned absolute
   - original sinusoidal encoding from *Attention Is All You Need*
   - RoPE
   - relative bias
   - ALiBi
   - RoPE scaling intuition
   - 2D positional encoding

## 项目特点

- 每个 notebook 只聚焦一个核心问题，不混主题。
- 解释优先级高于炫技，代码会尽量保留中间结果和 shape。
- 重要部分尽量给出最小实现，而不是只给定义。
- `utils/` 里的辅助函数也写成教学风格，docstring 和注释都偏详细。
- 训练部分和数据部分不是挂件，而是主线的一部分。
- 文件命名统一使用编号加下划线，不使用空格。

## 当前目录结构

```text
Build_DeepSeek_Step_by_Step/
├─ README.md
├─ PRD.md
├─ requirements.txt
├─ notebooks/
│  ├─ 01_python_and_matrix_foundations.ipynb
│  ├─ 02_tokenization_and_bpe.ipynb
│  ├─ 03_embeddings_and_language_model_inputs.ipynb
│  ├─ 04_self_attention_from_scratch.ipynb
│  ├─ 05_multi_head_attention.ipynb
│  ├─ 06_rope_and_position_encoding.ipynb
│  ├─ 07_rmsnorm_and_residual_connections.ipynb
│  ├─ 08_ffn_and_swiglu.ipynb
│  ├─ 09_build_a_basic_transformer_block.ipynb
│  ├─ 10_kv_cache_and_inference.ipynb
│  ├─ 11_mqa_and_gqa.ipynb
│  ├─ 12_mla_from_intuition_to_implementation.ipynb
│  ├─ 13_moe_routing_and_experts.ipynb
│  ├─ 14_pretraining_data_and_objective.ipynb
│  ├─ 15_sft_and_alignment_basics.ipynb
│  ├─ 16_reward_model_and_rl_intro.ipynb
│  ├─ 17_putting_everything_together.ipynb
│  ├─ 18_multi_stage_training_rlhf_ppo_grpo.ipynb
│  ├─ 19_data_collection_and_crawling.ipynb
│  ├─ 20_data_cleaning_filtering_and_dedup.ipynb
│  └─ 21_visualizing_position_encodings.ipynb
├─ utils/
│  ├─ tokenizer_utils.py
│  ├─ attention_utils.py
│  ├─ visualization_utils.py
│  └─ training_utils.py
└─ assets/
   ├─ figures/
   │  └─ README.md
   └─ sample_texts/
      ├─ tiny_corpus.txt
      ├─ instruction_examples.txt
      └─ raw_web_page_mock.html
```

## Notebook 路线

### Foundations

- `01_python_and_matrix_foundations.ipynb`
- `02_tokenization_and_bpe.ipynb`
- `03_embeddings_and_language_model_inputs.ipynb`
- `04_self_attention_from_scratch.ipynb`
- `05_multi_head_attention.ipynb`

### Core Transformer

- `06_rope_and_position_encoding.ipynb`
- `07_rmsnorm_and_residual_connections.ipynb`
- `08_ffn_and_swiglu.ipynb`
- `09_build_a_basic_transformer_block.ipynb`
- `10_kv_cache_and_inference.ipynb`

### Efficient Attention and Capacity

- `11_mqa_and_gqa.ipynb`
- `12_mla_from_intuition_to_implementation.ipynb`
- `13_moe_routing_and_experts.ipynb`

### Training and Alignment

- `14_pretraining_data_and_objective.ipynb`
- `15_sft_and_alignment_basics.ipynb`
- `16_reward_model_and_rl_intro.ipynb`
- `17_putting_everything_together.ipynb`
- `18_multi_stage_training_rlhf_ppo_grpo.ipynb`

### Data Pipeline

- `19_data_collection_and_crawling.ipynb`
- `20_data_cleaning_filtering_and_dedup.ipynb`

### Visualization

- `21_visualizing_position_encodings.ipynb`

## `utils/` 里有什么

`utils/` 不是简单占位，而是给 notebook 复用的教学辅助层：

- [tokenizer_utils.py](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/utils/tokenizer_utils.py)
  - 负责 tiny BPE、pair counting、merge replay、encode / decode
- [attention_utils.py](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/utils/attention_utils.py)
  - 负责 softmax、causal mask、single-head attention、head reshape、KV cache 直觉函数
- [visualization_utils.py](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/utils/visualization_utils.py)
  - 负责 heatmap、line trace、similarity matrix、figure 保存
- [training_utils.py](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/utils/training_utils.py)
  - 负责 cross entropy、masked SFT loss、pairwise reward loss、PPO、GRPO toy helpers

这些文件都偏教学风格，注释会故意写得比普通工具库更细。

## `assets/` 里有什么

`assets/` 用来放项目中可复用的静态资源。

- [assets/figures/README.md](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/assets/figures/README.md)
  - 说明图像资源应该如何归档
- [assets/sample_texts/tiny_corpus.txt](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/assets/sample_texts/tiny_corpus.txt)
  - 适合 BPE 和基础 tokenization notebook 使用
- [assets/sample_texts/instruction_examples.txt](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/assets/sample_texts/instruction_examples.txt)
  - 适合 SFT / instruction-format notebook 使用
- [assets/sample_texts/raw_web_page_mock.html](/Users/bizi/Desktop/GitHub/Build-DeepSeek-Step-by-Step/assets/sample_texts/raw_web_page_mock.html)
  - 适合 data collection / cleaning notebook 使用

## 环境

建议使用 Python 3.11。当前第一版 notebook 主要依赖：

- `numpy`
- `matplotlib`
- `torch`
- `jupyter`

这套依赖足够支撑项目里的最小实现、图像可视化和张量实验。

## 边界说明

当前版本仍然不追求：

- 完整工业级 DeepSeek 训练栈复刻
- 分布式训练系统实现
- CUDA / Triton / FlashAttention 内核优化
- 完整推理服务框架

这个项目更像一份可运行、可解释、可继续扩展的技术拆解笔记。重点是把从模块、到结构、到训练、到数据的完整链路讲清楚，而不是做仓库搬运或概念摘要。
