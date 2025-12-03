# 第三章：方法论与技术架构 (Methodology and Technical Architectures)

## 第一部分：正文草稿 (The Narrative Draft)

医学人工智能的技术基础经历了从传统机器学习到深度学习再到大语言模型的深刻演进。理解这一技术发展脉络对于把握当前医学AI研究的前沿态势至关重要。本章将系统性地阐述各类核心技术架构的原理、优势与局限，并深入分析它们在医学诊断中的具体应用表现。

### 3.1 传统机器学习方法

在深度学习兴起之前，传统机器学习方法在医学诊断领域占据主导地位。支持向量机（SVM）作为一种经典的监督学习算法，通过在高维特征空间中寻找最优分类超平面，在医学分类任务中展现出卓越的性能{REF_Asif_2025_MLDiagnostics}。多项研究表明，SVM在处理高维小样本医学数据集时表现出良好的泛化能力{REF_Velmurugan_2025_NDD}，这一特性对于标注数据稀缺的医学场景尤为重要{REF_Rahman_2024_SmartHealth}。然而，SVM的核函数选择和参数调优往往需要大量领域专家经验，这在一定程度上限制了其自动化应用潜力。

决策树与随机森林等集成学习方法为医学诊断提供了另一类重要工具。决策树以其天然的可解释性著称，其树状结构能够直观地展示诊断决策路径{REF_Asif_2025_MLDiagnostics}。随机森林通过集成多棵决策树并采用投票机制，有效降低了单一决策树易过拟合的风险{REF_Velmurugan_2025_NDD}{REF_Rahman_2024_SmartHealth}。在神经退行性疾病的早期筛查中，随机森林算法在处理多维生物标志物数据时展现出稳健的分类性能{REF_Velmurugan_2025_NDD}。与此同时，朴素贝叶斯分类器凭借其概率建模框架和对小样本数据的鲁棒性，在疾病风险评估任务中得到广泛应用{REF_Asif_2025_MLDiagnostics}。

值得注意的是，传统机器学习方法的性能在很大程度上依赖于人工设计的特征工程。这一特点既是其优势所在——领域专家可以将医学先验知识编码到特征中——也是其主要局限——特征设计的质量直接制约着模型的上限{REF_Rahman_2024_SmartHealth}。此外，传统方法在处理非结构化医学数据（如医学影像、临床文本）时面临显著挑战，这为深度学习方法的崛起创造了契机{REF_Asif_2025_MLDiagnostics}{REF_Zhou_2025_LLMDiagnosis}。

### 3.2 深度学习核心架构

#### 3.2.1 卷积神经网络

卷积神经网络（CNN）的出现标志着医学影像分析进入了自动特征学习的新时代。CNN通过卷积层实现对图像局部特征的自动提取，通过池化层实现空间降维和平移不变性，最终通过全连接层完成高层语义的分类决策{REF_CNNReview_2025_Medical}{REF_Rahman_2024_SmartHealth}。这种端到端的学习范式彻底颠覆了依赖人工特征工程的传统模式。

CNN架构的演进历程清晰地展示了深度学习在医学影像领域的发展轨迹。从AlexNet证明深度网络的可行性，到VGGNet探索网络深度与性能的关系，再到ResNet通过残差连接突破深度网络训练的瓶颈，每一代架构都为医学影像分析带来了新的突破{REF_CNNReview_2025_Medical}。DenseNet通过密集连接策略进一步增强了特征复用和梯度流动{REF_CNNReview_2025_Medical}{REF_Gou_2024_AIAssisted}。近年来，EfficientNet系列通过复合缩放策略，在计算效率和性能之间取得了更优的平衡{REF_CNNReview_2025_Medical}。

在医学图像分割领域，U-Net架构的提出具有里程碑意义。其编码器-解码器结构配合跳跃连接，能够有效融合低层细节信息和高层语义信息{REF_CNNReview_2025_Medical}{REF_Gou_2024_AIAssisted}。这一设计对于精确勾画病灶边界尤为重要，多项研究表明U-Net及其变体（如Attention U-Net、U-Net++）在器官分割和肿瘤检测任务中达到了接近人类专家的水平{REF_CNNReview_2025_Medical}{REF_Rahman_2024_SmartHealth}。然而，CNN在处理大尺度解剖结构时，受限于卷积核的局部感受野，难以有效建模长距离空间依赖{REF_Xiao_2025_LLMSurvey}，这一局限性推动了后续Transformer架构在医学影像中的引入。

#### 3.2.2 循环神经网络与长短期记忆网络

循环神经网络（RNN）及其改进版本长短期记忆网络（LSTM）为处理序列化医学数据提供了专门的技术方案。与CNN专注于空间特征提取不同，RNN通过循环结构维护隐藏状态，能够捕捉时间序列数据中的动态模式{REF_Nazi_2024_LLMHealthcare}{REF_Velmurugan_2025_NDD}。在电子健康档案（EHR）分析中，RNN能够学习患者就诊事件之间的时序关联，从而预测疾病发展轨迹{REF_Rahman_2024_SmartHealth}。

然而，标准RNN在处理长序列时面临梯度消失或爆炸的问题，这严重限制了其对长期依赖关系的建模能力{REF_Nazi_2024_LLMHealthcare}。LSTM通过引入门控机制（输入门、遗忘门、输出门）有效缓解了这一问题，使网络能够选择性地记忆或遗忘信息{REF_Nazi_2024_LLMHealthcare}{REF_Velmurugan_2025_NDD}。在神经退行性疾病的认知评估和行为分析中，LSTM被用于处理语音信号、步态数据等时间序列，以捕捉疾病进展的细微变化{REF_Velmurugan_2025_NDD}。尽管如此，RNN系列架构的串行计算特性使其难以充分利用现代并行计算硬件，训练效率相对较低{REF_Nazi_2024_LLMHealthcare}，这一局限性最终由Transformer架构所克服。

#### 3.2.3 Transformer与注意力机制

Transformer架构的诞生彻底重塑了序列建模的范式。其核心创新在于自注意力（Self-Attention）机制，该机制通过计算序列中任意两个位置之间的注意力权重，实现了对全局上下文的直接建模{REF_Xiao_2025_LLMSurvey}{REF_Nazi_2024_LLMHealthcare}。与RNN的循环依赖不同，自注意力允许完全并行计算，极大地提升了训练效率{REF_Nazi_2024_LLMHealthcare}{REF_Zhou_2025_LLMDiagnosis}。

在医学影像领域，Vision Transformer（ViT）通过将图像切分为补丁（Patch）并将其线性投影为向量序列，首次证明了Transformer在视觉任务中的有效性{REF_Xiao_2025_LLMSurvey}{REF_CNNReview_2025_Medical}。后续研究如Swin Transformer进一步引入了分层结构和移位窗口机制，在保持全局建模能力的同时降低了计算复杂度{REF_CNNReview_2025_Medical}。TransUNet等混合架构将CNN的局部特征提取优势与Transformer的全局建模能力相结合，在医学图像分割任务中取得了优异表现{REF_Xiao_2025_LLMSurvey}{REF_CNNReview_2025_Medical}。然而，Transformer的计算开销仍然显著高于传统CNN，尤其在处理高分辨率医学影像时，这一问题更为突出{REF_CNNReview_2025_Medical}{REF_Xiao_2025_LLMSurvey}。

### 3.3 大语言模型技术

#### 3.3.1 预训练语言模型与范式演进

大语言模型（LLM）的发展经历了从预训练语言模型（PLM）到生成式LLM的范式转换。BERT等编码器模型通过掩码语言建模（MLM）任务在大规模文本语料上进行预训练，学习到丰富的语言表征{REF_Nazi_2024_LLMHealthcare}{REF_Xiao_2025_LLMSurvey}。针对医学领域的特殊性，研究者开发了BioBERT、PubMedBERT和ClinicalBERT等领域特化模型，它们在生物医学文献或临床记录上进行继续预训练，以捕捉医学术语和临床话语的特殊语义{REF_Nazi_2024_LLMHealthcare}{REF_Xiao_2025_LLMSurvey}。

GPT系列模型的出现标志着生成式预训练范式的确立。这类解码器模型通过自回归方式预测下一词元，展现出强大的文本生成和涌现推理能力{REF_Xiao_2025_LLMSurvey}{REF_Zhou_2025_LLMDiagnosis}。随着模型规模的扩大，LLM展现出了"涌现"（Emergence）特性——即小模型不具备而大模型突然涌现的能力，如情境学习（In-Context Learning）和思维链推理（Chain-of-Thought）{REF_Xiao_2025_LLMSurvey}{REF_Nazi_2024_LLMHealthcare}。Med-PaLM系列在USMLE等医学考试中的表现表明，经过适当调整的LLM已具备接近人类专家的医学推理能力{REF_Xiao_2025_LLMSurvey}。

#### 3.3.2 多模态大语言模型

多模态大语言模型（MLLM）将视觉与语言能力进行融合，代表了医学AI发展的前沿方向。GPT-4V、Gemini和LLaVA等模型能够同时处理医学影像和临床文本，实现跨模态的理解与推理{REF_Xiao_2025_LLMSurvey}{REF_Zhou_2025_LLMDiagnosis}。在医学场景中，这种多模态能力对于整合影像学检查与病历信息具有重要价值{REF_Xiao_2025_LLMSurvey}{REF_Nazi_2024_LLMHealthcare}。

MLLM的构建通常采用"视觉编码器 + 投影层 + 语言模型"的架构范式{REF_Xiao_2025_LLMSurvey}。视觉编码器（如CLIP、ViT）负责提取图像特征，投影层将视觉特征映射到语言模型的嵌入空间，语言模型则完成最终的推理和生成{REF_Xiao_2025_LLMSurvey}{REF_Zhou_2025_LLMDiagnosis}。专门针对医学领域开发的LLaVA-Med、ChatCAD等模型，通过在医学影像-文本配对数据上进行微调，展现出对医学图像的专业理解能力{REF_Xiao_2025_LLMSurvey}。

#### 3.3.3 LLM应用技术

LLM在医学诊断中的应用涉及多种技术路径。提示工程（Prompt Engineering）通过精心设计的输入提示引导模型输出，包括零样本（Zero-shot）、少样本（Few-shot）和思维链（Chain-of-Thought, CoT）等策略{REF_Zhou_2025_LLMDiagnosis}{REF_Xiao_2025_LLMSurvey}。研究表明，CoT提示能够显著提升LLM在复杂医学推理任务中的表现{REF_Zhou_2025_LLMDiagnosis}。

检索增强生成（RAG）技术通过在推理时检索外部知识库，有效缓解了LLM的幻觉问题和知识时效性不足{REF_Zhou_2025_LLMDiagnosis}{REF_Nazi_2024_LLMHealthcare}。在医学诊断中，RAG可以动态检索最新的临床指南、药物信息或相似病例，为诊断推理提供可靠的证据支持{REF_Zhou_2025_LLMDiagnosis}。监督微调（SFT）和参数高效微调（PEFT，如LoRA）则提供了将通用LLM适配到特定医学任务的技术手段{REF_Xiao_2025_LLMSurvey}{REF_Zhou_2025_LLMDiagnosis}。人类反馈强化学习（RLHF）进一步通过与人类偏好对齐，增强模型输出的安全性和有用性{REF_Xiao_2025_LLMSurvey}。

### 3.4 可解释人工智能技术

深度学习模型的"黑箱"特性是其在临床应用中面临的核心障碍之一。可解释人工智能（XAI）旨在揭示模型决策背后的逻辑，从而增强临床用户对AI系统的信任{REF_Biswas_2024_XAI}{REF_CNNReview_2025_Medical}。从方法论角度，XAI技术可分为模型无关方法和模型特定方法两大类{REF_Biswas_2024_XAI}。

LIME（局部可解释模型无关解释）和SHAP（Shapley加性解释）是两种广泛应用的模型无关方法{REF_Biswas_2024_XAI}。LIME通过在输入空间的局部区域拟合可解释的代理模型来解释预测结果，而SHAP基于博弈论中的Shapley值为每个特征分配贡献度{REF_Biswas_2024_XAI}{REF_CNNReview_2025_Medical}。对于CNN模型，Grad-CAM及其变体通过可视化对最终预测贡献最大的图像区域，提供了直观的空间解释{REF_Biswas_2024_XAI}{REF_CNNReview_2025_Medical}。在医学影像诊断中，这类热力图可视化帮助临床医生理解AI关注的病灶区域，从而做出更为审慎的诊断决策{REF_Biswas_2024_XAI}。

然而，当前XAI技术仍面临诸多挑战。解释的忠实性（Faithfulness）——即解释是否真正反映了模型的实际决策过程——难以严格验证{REF_Biswas_2024_XAI}。此外，如何在解释的详尽性与用户可理解性之间取得平衡，以及如何将XAI方法扩展到LLM等更复杂的模型，仍是活跃的研究方向{REF_Biswas_2024_XAI}{REF_CNNReview_2025_Medical}。

### 3.5 多模态数据融合技术

医学诊断通常需要综合多种数据模态的信息，包括影像学检查、实验室指标、电子健康记录和基因组数据等{REF_Velmurugan_2025_NDD}{REF_Xiao_2025_LLMSurvey}。多模态数据融合技术旨在有效整合这些异构信息，以提升诊断的准确性和全面性。

根据融合发生的阶段，多模态融合策略可分为早期融合（Early Fusion）、晚期融合（Late Fusion）和中间融合（Intermediate Fusion）{REF_Velmurugan_2025_NDD}。早期融合在原始特征层面进行拼接，简单直接但可能引入噪声；晚期融合在各模态独立预测后进行决策级整合，保持了模态独立性但可能损失跨模态交互信息；中间融合在中间表征层面进行模态交互，在灵活性和表达能力之间取得平衡{REF_Velmurugan_2025_NDD}{REF_Xiao_2025_LLMSurvey}。

在神经退行性疾病诊断中，多模态融合展现出显著的优势。研究表明，整合MRI结构影像、PET功能代谢信息和认知评估量表数据，能够显著提升阿尔茨海默病及其前驱期（MCI）的诊断准确率{REF_Velmurugan_2025_NDD}。MLLM的兴起为多模态融合提供了新的技术路径，其端到端的学习方式可以自适应地学习跨模态对齐和交互{REF_Xiao_2025_LLMSurvey}{REF_Zhou_2025_LLMDiagnosis}。然而，不同模态数据的采集标准化、缺失值处理和隐私保护等问题，仍是多模态医学AI系统落地应用的现实挑战{REF_Velmurugan_2025_NDD}。

| **技术类别** | **代表性方法** | **核心优势** | **主要局限** | **医学应用场景** |
|------------|--------------|------------|------------|----------------|
| 传统机器学习 | SVM, RF, NB | 可解释性强、小样本适应 | 依赖人工特征工程 | 疾病风险筛查、生物标志物分析 |
| CNN | U-Net, ResNet, DenseNet | 自动特征学习、空间模式捕获 | 长距离依赖建模不足 | 医学影像分割与分类 |
| RNN/LSTM | LSTM, GRU | 序列依赖建模 | 长序列建模困难、并行效率低 | EHR时序分析、生理信号处理 |
| Transformer | ViT, Swin, TransUNet | 全局建模、高并行效率 | 计算开销大 | 高分辨率影像分析、跨模态学习 |
| LLM/MLLM | GPT-4, LLaVA-Med | 涌现推理、多模态融合 | 幻觉风险、计算资源需求高 | 临床问答、多模态诊断辅助 |
| XAI | LIME, SHAP, Grad-CAM | 增强模型透明度 | 解释忠实性验证困难 | 诊断决策解释与审计 |

---

## 第二部分：云端交互 JSON (The Cloud Interaction Layer)

```json
[[CLOUD_INTERACTION_LAYER]]
{
  "scope": "SECTION_DRAFT",
  "target_style": "IEEE_TMI",
  "language_check": "CHINESE_CONTENT",

  "verification_queue": [
    {
      "placeholder_id": "{REF_Asif_2025_MLDiagnostics}",
      "type": "VERIFY_PRIMARY",
      "source_file": "s11831-024-10148-w.pdf",
      "expected_metadata": {
        "authors": "S. Asif, Y. Wenhui, Saif-ur-Rehman, et al.",
        "title": "Advancements and Prospects of Machine Learning in Medical Diagnostics",
        "journal": "Archives of Computational Methods in Engineering",
        "year": 2025,
        "volume": 32
      },
      "instruction": "核实ML算法（SVM、决策树、朴素贝叶斯）介绍部分的页码范围"
    },
    {
      "placeholder_id": "{REF_Velmurugan_2025_NDD}",
      "type": "VERIFY_PRIMARY",
      "source_file": "wasj_7_6_403_PDF.pdf",
      "expected_metadata": {
        "authors": "Velmurugan et al.",
        "title": "AI and Multimodal Biomarkers in Neurodegenerative Disease",
        "journal": "World Academy of Sciences Journal",
        "year": 2025
      },
      "instruction": "核实神经退行性疾病ML方法和多模态融合部分的具体内容"
    },
    {
      "placeholder_id": "{REF_Rahman_2024_SmartHealth}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Machine learning and deep learning-based approach in smart healthcare.pdf",
      "expected_metadata": {
        "authors": "A. Rahman, T. Debnath, D. Kundu, et al.",
        "title": "Machine learning and deep learning-based approach in smart healthcare",
        "journal": "AIMS Public Health",
        "year": 2024,
        "volume": 11,
        "pages": "58-109"
      },
      "instruction": "核实深度学习基础架构介绍部分"
    },
    {
      "placeholder_id": "{REF_CNNReview_2025_Medical}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Deep Convolutional Neural Networks in Medical Image Analysis A Review (1).pdf",
      "expected_metadata": {
        "authors": "Authors from Information journal",
        "title": "Deep Convolutional Neural Networks in Medical Image Analysis: A Review",
        "journal": "Information",
        "year": 2025
      },
      "instruction": "核实CNN架构演进（AlexNet到EfficientNet）和U-Net描述"
    },
    {
      "placeholder_id": "{REF_Gou_2024_AIAssisted}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Research on Artificial-Intelligence-Assisted Medicine A Survey on Medical Artificial Intelligence (1).pdf",
      "expected_metadata": {
        "authors": "F. Gou, J. Liu, C. Xiao, et al.",
        "title": "Research on Artificial-Intelligence-Assisted Medicine: A Survey on Medical Artificial Intelligence",
        "journal": "Diagnostics",
        "year": 2024,
        "volume": 14,
        "article_number": 1472
      },
      "instruction": "核实U-Net医学影像应用部分"
    },
    {
      "placeholder_id": "{REF_Xiao_2025_LLMSurvey}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Xiao 等 - 2025 - A comprehensive survey of large language models and multimodal large language models in medicine.pdf",
      "expected_metadata": {
        "authors": "H. Xiao, F. Zhou, X. Liu, et al.",
        "title": "A comprehensive survey of large language models and multimodal large language models in medicine",
        "journal": "Information Fusion",
        "year": 2025,
        "volume": 117,
        "article_number": 102888
      },
      "instruction": "核实Transformer、LLM、MLLM技术架构部分"
    },
    {
      "placeholder_id": "{REF_Nazi_2024_LLMHealthcare}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Large Language Models in Healthcare and Medical Domain A Review.pdf",
      "expected_metadata": {
        "authors": "Z. A. Nazi, W. Peng",
        "title": "Large Language Models in Healthcare and Medical Domain: A Review",
        "journal": "Informatics",
        "year": 2024,
        "volume": 11,
        "article_number": 57
      },
      "instruction": "核实PLM到LLM发展历程、RNN局限性描述"
    },
    {
      "placeholder_id": "{REF_Zhou_2025_LLMDiagnosis}",
      "type": "VERIFY_PRIMARY",
      "source_file": "s44387-025-00011-z.pdf",
      "expected_metadata": {
        "authors": "S. Zhou, Z. Xu, M. Zhang, et al.",
        "title": "Large language models for disease diagnosis: a scoping review",
        "journal": "npj Digital Medicine (Nature Partner Journals)",
        "year": 2025
      },
      "instruction": "核实LLM诊断技术分类（提示工程、RAG、微调）"
    },
    {
      "placeholder_id": "{REF_Biswas_2024_XAI}",
      "type": "VERIFY_PRIMARY",
      "source_file": "Biswas - 2024 - A comprehensive review of explainable AI for disease diagnosis.pdf",
      "expected_metadata": {
        "authors": "A. A. Biswas",
        "title": "A comprehensive review of explainable AI for disease diagnosis",
        "journal": "Array",
        "year": 2024,
        "volume": 22,
        "article_number": 100345
      },
      "instruction": "核实XAI技术分类（LIME、SHAP、Grad-CAM）和挑战描述"
    }
  ],

  "search_requests": [
    {
      "placeholder_id": "{REQ_UNet_Ronneberger_2015}",
      "type": "SEARCH_SUPPORT",
      "keywords": ["U-Net", "Ronneberger", "2015", "MICCAI", "biomedical image segmentation"],
      "intent": "查找U-Net原始论文，为CNN分割部分提供原始引用支持"
    },
    {
      "placeholder_id": "{REQ_ResNet_He_2016}",
      "type": "SEARCH_SUPPORT",
      "keywords": ["ResNet", "He", "2016", "CVPR", "residual learning"],
      "intent": "查找ResNet原始论文，完善CNN架构演进的引用链"
    },
    {
      "placeholder_id": "{REQ_Transformer_Vaswani_2017}",
      "type": "SEARCH_SUPPORT",
      "keywords": ["Attention is all you need", "Transformer", "Vaswani", "2017", "NeurIPS"],
      "intent": "查找Transformer原始论文（已在Chapter 2查新，可复用结果）"
    },
    {
      "placeholder_id": "{REQ_ViT_Dosovitskiy_2021}",
      "type": "SEARCH_SUPPORT",
      "keywords": ["Vision Transformer", "ViT", "Dosovitskiy", "2021", "ICLR"],
      "intent": "查找ViT原始论文，为Transformer视觉应用提供支持"
    },
    {
      "placeholder_id": "{REQ_BERT_Devlin_2019}",
      "type": "SEARCH_SUPPORT",
      "keywords": ["BERT", "Devlin", "2019", "NAACL", "pre-training"],
      "intent": "查找BERT原始论文，完善PLM发展历程引用"
    }
  ]
}
[[END_INTERACTION]]
```

---

## 第三部分：参考文献预演 (Draft Bibliography - IEEE TMI Style)

### Draft References

[1] S. Asif, Y. Wenhui, Saif-ur-Rehman, Qurrat-ul-ain, K. Amjad, Y. Yueyang, S. Jinhai, and M. Awais, "Advancements and Prospects of Machine Learning in Medical Diagnostics: Unveiling the Future of Diagnostic Precision," *Arch. Comput. Methods Eng.*, vol. 32, pp. 853–883, 2025.

[2] Velmurugan *et al.*, "AI and Multimodal Biomarkers in Neurodegenerative Disease," *World Acad. Sci. J.*, vol. 7, art. no. 115, 2025.

[3] A. Rahman, T. Debnath, D. Kundu, M. S. I. Khan, A. A. Aishi, S. Sazzad, M. Sayduzzaman, and S. S. Band, "Machine learning and deep learning-based approach in smart healthcare: Recent advances, applications, challenges and opportunities," *AIMS Public Health*, vol. 11, no. 1, pp. 58–109, 2024.

[4] *Authors*, "Deep Convolutional Neural Networks in Medical Image Analysis: A Review," *Information*, vol. 16, art. no. 195, 2025.

[5] F. Gou, J. Liu, C. Xiao, *et al.*, "Research on Artificial-Intelligence-Assisted Medicine: A Survey on Medical Artificial Intelligence," *Diagnostics*, vol. 14, no. 14, art. no. 1472, 2024.

[6] H. Xiao, F. Zhou, X. Liu, T. Liu, Z. Li, X. Liu, and X. Huang, "A comprehensive survey of large language models and multimodal large language models in medicine," *Inf. Fusion*, vol. 117, art. no. 102888, 2025.

[7] Z. A. Nazi and W. Peng, "Large Language Models in Healthcare and Medical Domain: A Review," *Informatics*, vol. 11, no. 3, art. no. 57, Aug. 2024.

[8] S. Zhou, Z. Xu, M. Zhang, *et al.*, "Large language models for disease diagnosis: a scoping review," *npj Digit. Med.*, 2025.

[9] A. A. Biswas, "A comprehensive review of explainable AI for disease diagnosis," *Array*, vol. 22, art. no. 100345, 2024.

[10] {REQ_UNet_Ronneberger_2015} - O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *待查新*

[11] {REQ_ResNet_He_2016} - K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *待查新*

[12] {REQ_ViT_Dosovitskiy_2021} - A. Dosovitskiy *et al.*, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *待查新*

[13] {REQ_BERT_Devlin_2019} - J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *待查新*

---

**备注**：
- 本草稿严格遵循"深度叙述，拒绝碎片化"原则，全文采用中文学术段落形式撰写
- 遵循"拒绝孤证"原则，核心观点均由2-3篇文献共同支撑
- 对比表格为本章唯一允许的结构化展示
- 所有引用均进入云端交互层等待核实，参考文献保持英文原文（符合IEEE TMI标准）
