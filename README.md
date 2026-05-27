# Build_DeepSeek_Step_by_Step

`Build_DeepSeek_Step_by_Step` 是一个面向现代 LLM 核心模块的拆解与实现项目。目标不是复刻完整工业训练系统，而是把 tokenizer、embedding、attention、GQA、MLA、MoE、预训练与对齐这些关键部件逐层展开，用最小可运行实现和清晰的中间结果把它们的工作方式说明白。

这个项目关注三件事：

- 模块为什么存在
- 模块在计算图里的输入输出是什么
- 模块在工程上解决了什么成本或能力问题

## 项目范围

项目主线覆盖以下部分：

1. 文本如何被切成 token，以及 BPE 为什么有效。
2. token id 如何变成 embedding，位置如何注入模型。
3. self-attention、multi-head attention 与 causal mask 的核心计算过程。
4. RoPE、RMSNorm、SwiGLU、Transformer block 的基础结构。
5. KV cache、MQA、GQA、MLA 这些推理侧关键设计。
6. MoE 这种容量扩展路线的基本结构与路由逻辑。
7. DeepSeek-like 多阶段训练流程，从 pretraining 到 continued training、SFT、RLHF、PPO、GRPO 的整体链路。
8. 数据侧流程，包括语料爬取、原始数据整理、清洗、过滤、去重和质量控制。

## 项目特点

- 按模块拆解，每个 notebook 只处理一个核心问题。
- 所有文件统一使用编号加下划线命名，不使用空格。
- 每个 notebook 都包含问题定义、设计动机、核心公式、实现过程、结果检查和 trade-off 说明。
- 代码尽量保留中间张量、shape 和关键输出，避免黑盒封装。
- 内容重点放在结构理解与实现路径，不堆砌术语。
- 训练部分不只停留在 SFT，会把 reward model、RLHF、PPO、GRPO 和多轮迭代关系单独展开。
- 数据部分不只讲“有数据就能训”，会把爬取、抽取、清洗、过滤、去重这些前置环节拆开写。

## 目录结构

```text
Build_DeepSeek_Step_by_Step/
├─ PRD.md
├─ README.md
├─ requirements.txt
└─ notebooks/
   ├─ 01_python_and_matrix_foundations.ipynb
   ├─ 02_tokenization_and_bpe.ipynb
   ├─ 03_embeddings_and_language_model_inputs.ipynb
   ├─ 04_self_attention_from_scratch.ipynb
   ├─ 05_multi_head_attention.ipynb
   ├─ 06_rope_and_position_encoding.ipynb
   ├─ 07_rmsnorm_and_residual_connections.ipynb
   ├─ 08_ffn_and_swiglu.ipynb
   ├─ 09_build_a_basic_transformer_block.ipynb
   ├─ 10_kv_cache_and_inference.ipynb
   ├─ 11_mqa_and_gqa.ipynb
   ├─ 12_mla_from_intuition_to_implementation.ipynb
   ├─ 13_moe_routing_and_experts.ipynb
   ├─ 14_pretraining_data_and_objective.ipynb
   ├─ 15_sft_and_alignment_basics.ipynb
   ├─ 16_reward_model_and_rl_intro.ipynb
   ├─ 17_putting_everything_together.ipynb
   ├─ 18_multi_stage_training_rlhf_ppo_grpo.ipynb
   ├─ 19_data_collection_and_crawling.ipynb
   └─ 20_data_cleaning_filtering_and_dedup.ipynb
```

## Notebook 路线

### Foundations

- `01_python_and_matrix_foundations.ipynb`
- `02_tokenization_and_bpe.ipynb`
- `03_embeddings_and_language_model_inputs.ipynb`
- `04_self_attention_from_scratch.ipynb`
- `05_multi_head_attention.ipynb`

### Core_Transformer

- `06_rope_and_position_encoding.ipynb`
- `07_rmsnorm_and_residual_connections.ipynb`
- `08_ffn_and_swiglu.ipynb`
- `09_build_a_basic_transformer_block.ipynb`
- `10_kv_cache_and_inference.ipynb`

### Efficient_Attention_And_Capacity

- `11_mqa_and_gqa.ipynb`
- `12_mla_from_intuition_to_implementation.ipynb`
- `13_moe_routing_and_experts.ipynb`

### Training_And_Alignment

- `14_pretraining_data_and_objective.ipynb`
- `15_sft_and_alignment_basics.ipynb`
- `16_reward_model_and_rl_intro.ipynb`
- `17_putting_everything_together.ipynb`
- `18_multi_stage_training_rlhf_ppo_grpo.ipynb`

### Data_Pipeline

- `19_data_collection_and_crawling.ipynb`
- `20_data_cleaning_filtering_and_dedup.ipynb`

## 环境

建议使用 Python 3.11，基础依赖如下：

- `numpy`
- `matplotlib`
- `torch`
- `jupyter`

这些依赖足够支撑第一版 notebooks 的最小实现、简单可视化和张量实验。

## 边界说明

第一版不追求以下目标：

- 不复刻完整 DeepSeek 工业训练栈
- 不展开分布式训练系统
- 不覆盖 CUDA、Triton、FlashAttention 内核级优化
- 不构建完整推理服务框架

这套内容更接近一份从底层模块到整体架构、再到多阶段训练和数据流程的实现型技术笔记。重点是把核心部件拆开、说明白、跑起来，而不是做概念罗列或仓库搬运。
