---
layout:     post
title:      NLP Skim Reading
date:       2023-01-03
tags:   [nlp]
categories: 
- nlp
---

# A Frustratingly Easy Approach for Joint Entity and Relation Extraction

主要贡献：
- 设计了一个简单的 end2end 关系抽取，即采取 2 个独立的 encoder 分别用于实体抽取和关系识别，在 ACE04、ACE05 和 SciERC 数据集中达到 SOTA 结果，超过了之前的使用相同 pretrain encoders 的联合抽取模型(joint model)。
- 验证了分别学习实体和关系的**上下文表征**的重要性。
- 验证了在关系提取模型中整合实体信息（边界及类型）的重要性。
- 提出了一个高效的推理方法，在牺牲一点点性能（1% drop on f1）的情况下，推理速度有 8-16 倍的提升。

<div style="display: flex; justify-content: center;">
    <img src="https://file.ddot.cc/imagehost/2023/joint_entity_relation_model.png" width=678pt id='joint-entity-relation-model-image'>
</div>

Pipeline 模型：
1. 实体：如[上图(a)](#joint-entity-relation-model-image)所示，采取 span-level NER 的方式，即基于片段排列的方式，提取所有可能的片段排列，通过 Softmax 对每一个 Span 进行实体类型判断。Span 的好处是可以解决嵌套实体问题，但计算复杂度较高，理论上可能的实体组合有 $n(n+1)/2$ 个，其中 $n$ 为句子长度。
2. 关系：如上图(b)所示，对所有的实体 pair 进行关系分类。其中最重要的一点改进，就是将实体边界和类型作为标识符加入到实体 span 前后，然后作为关系模型的 input。例如，对于实体 pair（Subject和Object）可分别在其对应的实体前后插入以下标识符：
   - <S:Md>和</S:Md>：代表实体类型为 Method 的 Subject，S 是实体 span 的第一个 token， /S 是最后一个 token；
   - <O:Md>和</O:Md>：代表实体类型为 Method 的 Object，O 是实体 span 的第一个 token，/O 是最后一个 token； 


计算时，对每个实体 pair 中第一个 token 的编码进行组合，然后进行 Softmax 分类，判断是否属于某一个关系，计算开销比较大。 

> 文献参考
- span-based 的 NER 模型可参考 [《Span-based Joint Entity and Relation Extraction with Transformer Pre-training》](https://arxiv.org/pdf/1909.07755.pdf)
- span-based model 边界平滑，防止模型过于自信 [《Boundary Smoothing for Named Entity Recognition》](https://arxiv.org/pdf/2204.12031.pdf)
- 多头选择 [《Joint entity recognition and relation extraction as a multi-head selection problem》](https://www.sciencedirect.com/science/article/abs/pii/S095741741830455X?via%3Dihub)
- 层叠式指针标注 [《A Novel Cascade Binary Tagging Framework for Relational Triple Extraction》](https://arxiv.org/pdf/1909.03227.pdf)
  

## FAQ <span id='joint-relation-extract-faq'></span>
- Q: Joint model 跟 end2end model 有什么区别？

- Q: 有哪些可以推广的？

