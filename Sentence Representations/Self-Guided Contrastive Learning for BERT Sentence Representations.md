### Self-Guided Contrastive Learning for BERT Sentence Representations

###### Taeuk Kim, Kang Min Yoo, and Sang-goo Lee   $ACL 2021$

#### Abstract：  
&emsp;&emsp; BERT很强，但是如何使用BERT获得最好的句子嵌入呢？本文提出了一种**对比学习**方法，利用**自监督**的方式来提高BERT句子表征的质量。该方法使用self-supervised的方式对BERT进行fine-tune，需以来数据增强，并且可以使用[CLS] token作为句子嵌入向量。并且证明该方法具有高效率推理，并且对domain shift具有健壮性。

> （SentenceBert中提到使用[CLS]作为句子嵌入效果差，BERT-Whiting使用first-last-avg，而SimCSE则表示[CLS]和token avg效果差别不大）

#### 1. Introduction

&emsp;&emsp; 预训练Transformer推动了NLP的发展。但是他们的结果不能直接应用在句子层级的任务上，因为在预训练过程中，模型聚焦于token-level的任务，例如MLM。最典型的，使用BERT类模型作为句子编码器的方法，是在一个有监督的下游任务上进行fine-tune，例如NLI。通常使用最后一层的[CLS]作为整个句子的表示，这样做的好处在于，在fine-tune过程中，只有[CLS]在BERT和下游任务之间进行交互，帮助它捕获句子整体的信息。但是在没有监督训练数据时，如何进行fine-tune？最通常的做法是对BERT最后一层的token embedding做pooling。本文对使用使用哪些层、池化方法做了分析，在STS数据集上进行了测试，结果表明他们都很烂。

&emsp;&emsp; 该方法的核心思想是，利用BERT中间层的hidden representation作为positive samples，并且要求最终的句子嵌入应该和中间层的表示接近。本方法不适用庶几乎增强，因此比大部分其他对比学习的方法都简单，并且将CV领域NTXent loss针对BERT做了一些修改。

#### 2. Related Work

##### Contrastive Representation Learning   

&emsp; &emsp; 最早可以追溯到word2vec的负采样，之后有使用对比学习目标来训练Transformer类模型的，和本文类似。但是本文并不打算重新训练一个预训练模型，而是在现有预训练模型上进行修改。

##### Fine-tuning BERT with Supervision  
&emsp;&emsp; BERT fine-tune不总是有用，尤其是缺乏目标域的数据时。有人提出使用有监督对比学习目标作为fine-tune过程中的辅助手段。作为对比，我们来解决在没有有监督数据时，要如何完成该任务。

##### Sentence Embedding from BERT  

- SBERT：孪生网络共享参数，使用最后一层使用mean pooling而得到定长句子嵌入，在NLI数据集上进行fine-tune。

#### 3. Method

&emsp;&emsp; 类似于sbert，使用监督方式，需要大量有监督数据。对比学习，通常要使用数据增强手段，而本文的方法只需要原始的无监督数据，不需要额外数据增强，直接使用BERT中间层的隐状态。

##### 3.1 Contrastive Learning

&emsp;&emsp; 一种不需要外部数据的对比学习方法是虚拟对抗训练，但不能保证句子嵌入的语义在加入随机扰动后是否会发生改变，作为替代，使用BERT中间层的隐状态，这样能够从概念上保证两个向量的语义相似，本文称这种方法为 *self-guided contrastive learning*.

