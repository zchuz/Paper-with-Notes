### 有监督, 对比学习, 句子表征
#### Pairwise Supervised Contra Learning of Sentences Representation - Amazon AI
##### Abstract
&emsp;&emsp; 近期成功的句子表征学习再有监督的情况下, 通常在NLI数据集上使用triplet loss或siamese loss进行微调.  他们都存在着一个问题, 就是矛盾句子对中的句子不一定是来自于同一个语义范畴的, 因此仅仅优化蕴含和矛盾是不足以捕获高级语义结构的.(讲故事) 使用NLI有监督训练的缺点是, 他们只考虑了单个句子对, 收到局部最优的影响. 本文提出了**PairSupCom**, 基于实例判别的方法, 希望将通过高层次范畴的概念编码将语义蕴含, 矛盾联系起来. 
##### 1. Introduction
&emsp;&emsp; 介绍NLI数据集, 本文与SentenceBert进行对比. 本文收到自监督学习的启发, 提出了实例鉴别损失+句子对语义推理的联合优化方式. 实例鉴别学习可以在向量空间中将相似的实例隐式的分组在一起, 无需显示的学习. 现有的sota方法在STS数据集上表现出色, 但是在嵌入的categorical semantic structure(文本聚类)上会有所欠缺, 更好的捕获高级语义概念反过来可以促进因好矛盾的推理. **简而言之, 现有的sota方法在STS上性能优秀, 但是在文本聚类任务上表现变差, 作者认为折是嵌入空间产生了退化**
##### 2. Related Work
1. 通过NLI数据集学习句子表征, 例如InferSent, Universal SE, SBERT.
2. Self-Supervised Instance Discrimination: 对比学习实质上是解决了在一个mini-bathc中, 在茫茫多的negative pair中判别出positive pair.
3. Deep Metric Learning: 使用三元组损失或孪生损失, 通常需要大量的训练数据. 通常由两种解决方案, 一种解决方案是在训练数据中采样hard positive/negative, 这种方案需要大量人力; 另一种方案是在triplet loss上加入对比损失, 让正例能够对抗反例.  本文的工作结合了上述两种方法, 区别在于上述的方法需要类别层次的监督来选择负样本(聚类), **作者希望找到强反例, 这些反例比一眼就能看出来的反例更加有价值**. 但是NLI数据集中没有类别的标签, 于是使用contradict作为hard-negative需要假设hard-negative和anchor他们处于相同的语境中, 并且语义相似. 4.3中探讨了这一假设.

##### 3. Model
&emsp;&emsp; 作者使用SNLI+MNLI作为训练数据.
