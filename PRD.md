# Build_DeepSeek_Step_by_Step_PRD

## 1. 项目名称

Build_DeepSeek_Step_by_Step

## 2. 项目定位

这是一个围绕现代 LLM 关键结构、训练链路和数据链路展开的拆解与实现项目。项目重点不是复刻完整工业级 DeepSeek 训练栈，而是把一条典型的 DeepSeek-like 技术路线拆成清晰的 notebook、辅助工具和图像资产，让读者可以逐层理解：

- tokenizer 如何把原始文本转成 token
- embedding、position encoding、RoPE 如何组织输入表示
- self-attention、MHA、MQA、GQA、MLA 如何在不同层级上解决计算与缓存问题
- FFN、SwiGLU、MoE 如何承担表达能力与容量扩展
- pretraining、continued training、SFT、reward model、RLHF、PPO、GRPO 如何组成完整训练主线
- 数据爬取、正文抽取、清洗、过滤、去重、质量分桶如何影响模型底座
- 位置编码为什么不仅要讲公式，还要讲可视化图和几何直觉

## 3. 项目目标

### 3.1 技术目标

项目完成后，应形成一套可以顺序阅读、也可按主题跳读的 notebook 与辅助代码资产，使读者能够：

1. 追踪从文本到 token ids、再到 embedding 的输入链路。
2. 从零实现基础 self-attention 与 multi-head attention。
3. 理解 decoder-only Transformer block 的核心数据流。
4. 区分 MHA、MQA、GQA、MLA 在缓存、带宽和推理效率上的差异。
5. 理解 MoE 的路由逻辑、activated parameters 与容量扩展方式。
6. 理解从 pretraining 到 PPO / GRPO 的多阶段训练主线。
7. 理解从数据采集到训练语料落地的基本工程链路。
8. 通过热力图、相似度图和曲线图直观看懂不同位置编码的差异。

### 3.2 交付目标

项目应满足以下要求：

- 每个 notebook 只聚焦一个核心问题。
- 所有 notebook 使用 `01_xxx.ipynb`、`02_xxx.ipynb` 这种编号加下划线命名。
- 每个 notebook 都包含问题定义、设计动机、核心公式、实现过程和结果检查。
- 代码注释必须说明关键 shape、张量角色和重要中间结果。
- 文本表达以技术拆解、实现说明、架构分析为主，不采用课堂讲义口吻。
- `utils/` 目录中的工具代码也必须是教学风格，而不是黑盒工具箱。
- `assets/` 必须提供可以支撑 notebook 的样本文本和图像资产目录。

## 4. 目标读者

核心读者包括：

- 想系统理解现代 LLM 结构的工程师
- 想从论文术语走到实现细节的开发者
- 想把 BPE、Attention、GQA、MLA、MoE、RLHF 放进统一认知框架的人

读者预期具备：

- 基本 Python 能力
- 对矩阵乘法和基础概率有粗略概念
- 愿意通过最小实现追踪中间张量、数据流和结构差异

## 5. 非目标

当前版本默认不覆盖以下内容：

- 完整工业训练集群与分布式训练工程
- CUDA、Triton、FlashAttention 等内核级优化
- 大规模调度系统和完整服务部署栈
- 对某个官方仓库做逐文件级的一比一复刻

这些内容可以作为后续扩展，但不属于当前主线交付范围。

## 6. 核心原则

### 6.1 单模块聚焦

每个 notebook 解决一个明确问题，避免多个主题混杂。

### 6.2 先问题，后机制，再实现

每个模块都应先说明它在系统里解决什么问题，再说明结构与公式，最后给出最小实现。

### 6.3 Shape 必须透明

所有关键张量都需要明确 shape 和语义，避免只给公式不说明数据形状。

### 6.4 先基础实现，再引入变体

例如先给出单头 attention，再到 MHA；先解释 MHA，再到 MQA / GQA；先说明 KV cache，再说明 MLA。

### 6.5 Notebook 和工具文件都必须可独立阅读

除了 notebook 本身，`utils/` 里的代码也应具备教学价值。函数 docstring 需要足够清楚，使读者即使单独打开工具文件，也能看懂它服务于哪一节 notebook。

## 7. 交付形式

主要交付物包括：

- 一组按顺序组织的 Jupyter notebooks
- 一个与 notebook 配套的 `utils/` 目录
- 一个存放样本文本和图像资源的 `assets/` 目录
- 项目说明与规划文档 `README.md`、`PRD.md`

## 8. 当前目录结构

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

## 9. 模块路线

### 9.1 Foundations

- `01_python_and_matrix_foundations.ipynb`
- `02_tokenization_and_bpe.ipynb`
- `03_embeddings_and_language_model_inputs.ipynb`
- `04_self_attention_from_scratch.ipynb`
- `05_multi_head_attention.ipynb`

### 9.2 Core Transformer

- `06_rope_and_position_encoding.ipynb`
- `07_rmsnorm_and_residual_connections.ipynb`
- `08_ffn_and_swiglu.ipynb`
- `09_build_a_basic_transformer_block.ipynb`
- `10_kv_cache_and_inference.ipynb`

### 9.3 Efficient Attention and Capacity

- `11_mqa_and_gqa.ipynb`
- `12_mla_from_intuition_to_implementation.ipynb`
- `13_moe_routing_and_experts.ipynb`

### 9.4 Training and Alignment

- `14_pretraining_data_and_objective.ipynb`
- `15_sft_and_alignment_basics.ipynb`
- `16_reward_model_and_rl_intro.ipynb`
- `17_putting_everything_together.ipynb`
- `18_multi_stage_training_rlhf_ppo_grpo.ipynb`

### 9.5 Data Pipeline

- `19_data_collection_and_crawling.ipynb`
- `20_data_cleaning_filtering_and_dedup.ipynb`

### 9.6 Visualization and Inspection

- `21_visualizing_position_encodings.ipynb`

## 10. 详细模块规划

### 10.1 `01_python_and_matrix_foundations.ipynb`

目标：
建立后续所有模块共享的张量、矩阵、batch、softmax、mask 基础。

产出：

- 向量、矩阵、batch、sequence length 的最小说明
- softmax 和 causal mask 的最小实现
- 一个最小 next-token 预测玩具例子

### 10.2 `02_tokenization_and_bpe.ipynb`

目标：
拆解 tokenizer 的输入输出路径，并实现最小 BPE。

产出：

- BPE 训练流程说明
- merge 规则生成过程
- encode / decode 最小实现
- vocab_size、压缩率和泛化能力的关系说明

### 10.3 `03_embeddings_and_language_model_inputs.ipynb`

目标：
说明 token ids 如何进入模型并转成连续表示。

产出：

- embedding table 与 lookup 的最小实现
- 输入 shape 说明
- token ids 到 embedding vectors 的完整路径

### 10.4 `04_self_attention_from_scratch.ipynb`

目标：
从零说明 self-attention 的分数计算、mask 和输出聚合过程。

产出：

- Q / K / V 的角色说明
- 单头 self-attention 实现
- attention weights 打印与解释
- `O(n^2)` 成本来源说明

### 10.5 `05_multi_head_attention.ipynb`

目标：
说明多头机制如何扩展 attention 的表达能力。

产出：

- head splitting 与 concat 的最小实现
- 不同 head 的权重对比
- 多头结构与维度分配说明

### 10.6 `06_rope_and_position_encoding.ipynb`

目标：
说明 attention 为什么需要位置编码，以及 RoPE 如何作用到 Q / K。

产出：

- 绝对位置编码与相对位置编码的最小比较
- RoPE 最小实现
- 位置变化前后的数值对比

### 10.7 `07_rmsnorm_and_residual_connections.ipynb`

目标：
说明 residual 与 RMSNorm 在稳定深层 Transformer 上的作用。

产出：

- LayerNorm 与 RMSNorm 对比
- residual 路径最小示例
- pre-norm 结构说明

### 10.8 `08_ffn_and_swiglu.ipynb`

目标：
说明 FFN 在 block 中的角色，以及 SwiGLU 的门控差异。

产出：

- 基础 FFN 最小实现
- SwiGLU 版本实现
- 逐 token 非线性变换说明

### 10.9 `09_build_a_basic_transformer_block.ipynb`

目标：
把前面模块拼成一个最小 decoder-only Transformer block。

产出：

- block 数据流说明
- norm、attention、residual、FFN 的串接实现
- 前向传播最小示例

### 10.10 `10_kv_cache_and_inference.ipynb`

目标：
说明推理阶段 KV cache 如何减少重复计算。

产出：

- 无 cache 与有 cache 的计算对比
- K / V 缓存增长方式说明
- 自回归生成里的重复劳动分析

### 10.11 `11_mqa_and_gqa.ipynb`

目标：
说明 MHA、MQA、GQA 的结构差异与工程权衡。

产出：

- 三种结构的形状对比
- K / V 共享方式说明
- cache 成本对比

### 10.12 `12_mla_from_intuition_to_implementation.ipynb`

目标：
说明 MLA 如何通过更紧凑的状态表示继续压缩推理成本。

产出：

- latent compression 思路
- 压缩表示到 K / V 空间的映射过程
- 原始 K / V cache 与 latent cache 的成本对比

### 10.13 `13_moe_routing_and_experts.ipynb`

目标：
说明 MoE 为什么扩大总容量，以及 router、top-k、load balancing 在系统里分别做什么。

产出：

- dense FFN 与 MoE 的差异
- router 的最小实现
- top-k expert 选择逻辑
- total parameters 与 activated parameters 的区别

### 10.14 `14_pretraining_data_and_objective.ipynb`

目标：
说明 pretraining 在完整训练链里的职责，以及 next-token prediction 为什么足以形成强 base model。

产出：

- token-level cross entropy 解释
- teacher forcing 最小示例
- token budget、context length、batch size 区分
- pretraining 在全链路中的位置说明

### 10.15 `15_sft_and_alignment_basics.ipynb`

目标：
说明 SFT 如何把 base model 推向 assistant-style 行为分布。

产出：

- assistant-only loss mask 解释
- 指令格式数据示例
- SFT 与 pretraining 的目标差异
- 多轮对话监督最小示例

### 10.16 `16_reward_model_and_rl_intro.ipynb`

目标：
说明 reward model、pairwise preference、RLHF、DPO 的相对位置。

产出：

- pairwise reward loss 公式
- reward model scoring 示例
- RLHF 与 DPO 的最小直觉
- 偏好优化与知识学习的边界说明

### 10.17 `17_putting_everything_together.ipynb`

目标：
把结构线、训练线和数据线重新合成一张统一地图。

产出：

- 从文本到输出的全链路回顾
- 高级模块在系统中的位置说明
- DeepSeek-like 闭环全景图

### 10.18 `18_multi_stage_training_rlhf_ppo_grpo.ipynb`

目标：
系统讲清楚多阶段训练链路，以及 PPO、GRPO 在最后阶段的作用差异。

产出：

- pretraining -> continued training -> SFT -> reward model -> PPO / DPO / GRPO 链路说明
- checkpoint lineage 示例
- PPO clip 目标与 KL 约束解释
- GRPO 的组内相对优势示例

### 10.19 `19_data_collection_and_crawling.ipynb`

目标：
说明训练语料最初是如何从网页、代码、论文、论坛等来源被采集出来的。

产出：

- source registry 设计
- crawl pipeline 结构
- 正文抽取示意
- metadata 保留策略
- frontier / revisit policy 直觉

### 10.20 `20_data_cleaning_filtering_and_dedup.ipynb`

目标：
说明原始记录怎样一步步变成可训练文本。

产出：

- normalization
- quality filtering
- exact dedup
- near-duplicate detection
- quality bucketing

### 10.21 `21_visualizing_position_encodings.ipynb`

目标：
用图像和公式一起解释不同位置编码的结构差异，而不是只停留在概念定义。

产出：

- learned absolute positional embeddings 热力图
- original sinusoidal positional encoding 公式与热力图
- RoPE 几何旋转与相似度图
- relative bias / ALiBi 可视化
- RoPE scaling 直觉图
- 2D positional encoding 图

## 11. `utils/` 的职责

### 11.1 `utils/tokenizer_utils.py`

负责提供 BPE 和 tokenization notebook 可复用的教学辅助函数。

包括：

- 字符级切分
- pair counting
- merge replay
- tiny BPE training
- encode / decode

### 11.2 `utils/attention_utils.py`

负责提供 attention notebook 的可复用函数。

包括：

- stable softmax
- causal mask
- masked score application
- single-head self-attention
- head split / combine
- KV cache 直觉函数

### 11.3 `utils/visualization_utils.py`

负责提供可复用图像辅助函数。

包括：

- heatmap
- line traces
- cosine similarity matrix
- figure saving

### 11.4 `utils/training_utils.py`

负责提供训练 notebook 的可复用目标函数和 toy helper。

包括：

- cross entropy
- masked SFT loss
- pairwise reward loss
- PPO clipped objective
- GRPO group-relative normalization

## 12. `assets/` 的职责

### 12.1 `assets/figures/`

用于存放长期保留的导出图，而不是临时 notebook 执行结果。

### 12.2 `assets/sample_texts/`

用于存放 notebook 里会复用的小语料和 toy 数据样本。

当前建议内容包括：

- tiny corpus
- instruction examples
- raw web page mock

## 13. 验收标准

如果项目达到以下标准，则认为当前阶段成功：

1. `01` 到 `21` notebook 全部可顺序阅读，并且主题边界清楚。
2. `README.md` 与 `PRD.md` 和当前实际项目状态一致。
3. `utils/` 不是空壳，而是真正能被 notebook 调用的教学辅助代码。
4. `assets/` 提供基础图像目录和样本文本，不是空目录。
5. 训练链、数据链和位置编码可视化已经进入主线，而不是临时补充。
6. 代码注释足够详细，读者单看工具文件也能看懂函数在做什么。

目标：
说明 MoE 如何通过路由和稀疏激活扩展容量。

产出：

- router 与 top-k 逻辑
- expert 路径最小实现
- activated parameters 与 total parameters 区分

### 10.14 `14_pretraining_data_and_objective.ipynb`

目标：
说明预训练阶段的 next-token objective 和 cross entropy loss。

产出：

- logits 到概率到 loss 的最小链路
- batch、context length、token 数的基本说明
- 预训练语料规模的作用说明

### 10.15 `15_sft_and_alignment_basics.ipynb`

目标：
说明 SFT 在整体对齐流程中的位置。

产出：

- prompt / response 数据格式示例
- loss mask 思路
- SFT 与 pretraining 的关系说明

### 10.16 `16_reward_model_and_rl_intro.ipynb`

目标：
说明奖励模型、偏好数据和偏好优化的基本位置。

产出：

- pairwise preference 最小示例
- reward model 打分直觉
- PPO / DPO 在流程图中的位置说明

### 10.17 `17_putting_everything_together.ipynb`

目标：
把所有模块重新放回一个统一的架构视角。

产出：

- 从文本到 token、从 token 到 block、从 block 到训练流程的总图
- attention 变体、容量扩展和对齐流程的统一概览
- 下一阶段扩展方向

### 10.18 `18_multi_stage_training_rlhf_ppo_grpo.ipynb`

目标：
说明 DeepSeek-like 模型在结构之外，如何通过多阶段训练逐步获得基础能力、指令跟随能力和偏好对齐能力。

产出：

- 从 pretraining、continued training、SFT 到 RLHF 的完整流程图
- reward model、RLHF、PPO、GRPO 的角色说明
- PPO 和 GRPO 的最小示例与差异分析
- 多轮训练中数据源、模型版本、反馈信号如何迭代

### 10.19 `19_data_collection_and_crawling.ipynb`

目标：
说明预训练与对齐数据最开始从哪里来，原始语料如何采集，以及不同来源的特点与风险。

产出：

- 常见语料来源说明：网页、代码、论文、问答、论坛、文档
- 爬取与采集的最小流程说明
- 站点结构、robots、频控、格式抽取、正文抽取等基础问题
- 原始数据存储格式示例

### 10.20 `20_data_cleaning_filtering_and_dedup.ipynb`

目标：
说明原始抓取数据为什么不能直接喂给模型，以及清洗、过滤、去重和质量控制如何影响训练效果。

产出：

- HTML 清理、语言识别、长度过滤、质量过滤的最小流程
- 去重与近重复检测的基本思路
- 噪声数据、低质量数据、模板污染数据的处理方式
- 数据质量与预训练结果之间的关系说明

## 11. Notebook 标准模板

每个 notebook 统一采用以下结构：

- 标题
- Problem
- Dependencies
- Module_Goals
- Context_And_Motivation
- Core_Idea
- Math
- Implementation
- Inspection
- Trade_Offs
- Summary
- Next_Module

## 12. 文风要求

- 清晰、直接、结构化
- 解释具体，不用空泛结论占篇幅
- 先定义问题，再说明结构，再落到实现
- 保留必要的直觉说明，但整体口吻以技术说明为主
- 代码注释服务于理解 shape、数据流和工程意义

## 13. 代码要求

### 13.1 风格

- 优先使用简单直接的 Python
- 避免不必要的重封装
- 尽量保留中间变量与打印结果
- 保证最小实现可读、可运行、可检查

### 13.2 注释

- 关键张量必须标注 shape
- 关键步骤前需要说明意图
- 复杂公式需要映射回代码
- 注释不重复代码表面语义

### 13.3 优先级

1. 清晰
2. 正确
3. 可运行
4. 再考虑抽象与复用

## 14. 验收标准

第一版完成后，应满足：

1. 17 个 notebook 命名规范统一，全部使用下划线。
2. BPE、Attention、GQA、MLA、MoE 都有最小实现与结构说明。
3. 每个 notebook 都能独立说明本模块的问题、机制、实现与代价。
4. 文本表达不采用课堂化模板，整体更像技术拆解文档。
5. 项目整体从基础模块到系统视角衔接自然。
6. 训练链路明确覆盖 RLHF、PPO、GRPO 和多阶段迭代逻辑。
7. 数据链路明确覆盖爬取、清洗、过滤和去重。

## 15. 优先级

### P0

- `01_python_and_matrix_foundations.ipynb`
- `02_tokenization_and_bpe.ipynb`
- `03_embeddings_and_language_model_inputs.ipynb`
- `04_self_attention_from_scratch.ipynb`
- `05_multi_head_attention.ipynb`
- `06_rope_and_position_encoding.ipynb`
- `09_build_a_basic_transformer_block.ipynb`
- `11_mqa_and_gqa.ipynb`
- `12_mla_from_intuition_to_implementation.ipynb`

### P1

- `07_rmsnorm_and_residual_connections.ipynb`
- `08_ffn_and_swiglu.ipynb`
- `10_kv_cache_and_inference.ipynb`
- `13_moe_routing_and_experts.ipynb`

### P2

- `14_pretraining_data_and_objective.ipynb`
- `15_sft_and_alignment_basics.ipynb`
- `16_reward_model_and_rl_intro.ipynb`
- `17_putting_everything_together.ipynb`
- `18_multi_stage_training_rlhf_ppo_grpo.ipynb`
- `19_data_collection_and_crawling.ipynb`
- `20_data_cleaning_filtering_and_dedup.ipynb`

## 16. 后续扩展

- FlashAttention 最小说明与对比
- tiny language model 训练闭环
- sampling：greedy、temperature、top-k、top-p
- 长上下文与 RoPE scaling
- quantization 基础
- distributed training 概念图
- vLLM / paged attention 背后的状态管理思路
- 数据标注、偏好数据构造和自动数据合成

## 17. 结论

`Build_DeepSeek_Step_by_Step` 应该呈现为一套从底层模块到系统结构的实现型技术项目，而不是概念罗列或仓库摘抄。它的核心价值在于：把关键模块拆开、把结构差异写清、把最小实现落地，并让整个 DeepSeek-like 架构的主线变得可追踪、可解释、可继续扩展。
