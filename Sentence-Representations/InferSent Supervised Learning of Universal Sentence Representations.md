#### Supervised Learning of Universal Sentence Representation from NLI Data

###### Alexis Connuau, Douwe Kiela, Holger Schwenk    $\mathbb {EMNLP 2018} $

##### Abstract:

&emsp;&emsp; **这篇文章很老, 已经过时, 只是简单的看一下**. 本文使用**SNLI**有监督数据, 来提高句子嵌入的表现.

##### 1. Introduction:

&emsp;&emsp; 本文将在大规模语料上学习句子表示, 并且在其他任务中使用这些句子嵌入表示. 本文使用了预训练的模型(非Transformer, 这篇文章17年的, 还没有BERT), 包括卷积模型, 序列模型等. 实验结果发现bi-LSTM with max pooling的效果是最好的.

##### 3. Approach

&emsp;&emsp; 介绍如何使用NLI数据集进行训练, 之后描述句子编码器的模型结构. 对序列循环模型, 卷积模型, 注意力模型都进行了实验测试.

##### 3.1 NLI Task

&emsp;&emsp; NLI的训练通常有两种方式, 双塔式和交互式. 为了通用性, 选择了双塔式.

> 双塔式: 类似于SentenceBert, 两个输入分别输入Encoder, 在编码结束之后再进行交互.
>
> 交互式: 类似于Bert的NSP任务, 两个输入同时输入到一个Encoder, 并在其中进行交互.

##### 3.2 Sentence Encoder Architecture

&emsp;&emsp; 将不定长的句子转化为定长的稠密向量表示. 本文尝试了七种方法, 分别是*LSTM, GRU, Bi-GRU-concat-last-hidden, Bi-LSTM with Pooling, self-attention, hierarchical Conv*

##### 没有看下去的必要了 Bi-LSTM + maxpooling 在SNLI上进行训练

