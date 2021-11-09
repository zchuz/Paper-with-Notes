#### Universal Sentence Encoder

###### Google Research   $\mathbb {2018}$

##### Abstract

&emsp;&emsp; 本文提出了句子向量编码器, 可以将得到的句子嵌入应用到其他任务. 该编码器具有两个变体, 本别是速度和准确率的权衡.

##### 1. Introduction

&emsp;&emsp; 现有NLP缺乏监督数据. 本文提出了两个嵌入模型, 并且就训练数据大小以及任务表现进行了比较.

##### 3. Encoder

&emsp;&emsp;本文的两种编码器有着不同的设计目标, 一种基于transformer架构, 目标是高精度但高开销; 另一种是DAN, 低开销但是降低了精度.

##### 3.1 Transformer

&emsp;&emsp;使用Transformer的Encoder部分来获得句子嵌入, 对每个位置(token)的向量进行平均得到最终的嵌入表示.

##### 3.2 DAN

&emsp;&emsp; 使用DAN, 即将词嵌入平均之后再送入多层MLP中, 得到固定长度的句子嵌入表示. Transformer和DAN都使用PTB tokenizer 来对句子进行分词.

##### 3.3 Encoder Training Data

&emsp;&emsp;无监督数据来自于互联网, 类似于wiki. 有监督数据来自于SNLI.

接下来就是用得到的嵌入在多个任务上进行测试 很无聊的文章.