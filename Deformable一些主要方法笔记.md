# Deformable一些主要方法笔记

**i). Multi-Scale Features & Scale-Level Embedding**

在上一节也提到过，作者说可变形注意力能够用于处理多尺度特征，于是在backbone部分提取了不同层级的特征，总共有4层，其中C3~C5来自ResNet，下采样率对应为8、16、32，C6由C5经过步长为2的3x3卷积得到。

![img](https://pic3.zhimg.com/80/v2-d77875e46cc5190b45abb5afabdfb09e_1440w.jpg)

要知道，DETR仅用了单尺度特征，于是对于特征点位置信息的编码，使用的是三角函数，不同位置的特征点会对应不同的编码值，没问题。

但是，注意了，这仅能区分位于单尺度特征点的位置！而在多尺度特征中，位于不同特征层的特征点可能拥有相同的(h,w)坐标，这样就无法区分它们的位置编码了。

针对这个问题，作者增加使用一个称之为'scale-level embedding'的东东，它仅用于区分不同的特征层，也就是同一特征层中的所有特征点会对应相同的scale-level embedding，于是有几层特征就使用几个不同的scale-level embedding。

另外，不同于三角函数那种固定地利用公式计算出来的编码方式，这个scale-level embedding是随机初始化并且是随网络一起训练的、是可学习的：

![img](https://pic1.zhimg.com/80/v2-534f3176f39e833433e32744edec7428_1440w.jpg)

在实际使用时，这个 scale-level embedding 与基于三角函数公式计算的 position embedding 相加在一起作为位置信息的嵌入：

![img](https://pic1.zhimg.com/80/v2-f6110e9803accc8f45a5d861711aaa14_1440w.jpg)

**ii). Deformable Attention(& Multi-Scale)**

可变形注意力的道理用大白话来说很简单：query不是和全局每个位置的key都计算注意力权重，而是对于每个query，仅在全局位置中采样部分位置的key，并且value也是基于这些位置进行采样插值得到的，最后将这个局部&稀疏的注意力权重施加在对应的value上。

OK，“普通话”讲完，该高大上（公式+代码）一番了。Transformer中多头注意力的公式如下：

![img](https://pic1.zhimg.com/80/v2-f13b43439fba35ef82eff63b07fc7b50_1440w.png)

其中，zq看作query，由x经过线性变换生成，q是对应的索引，k是key的索引，Omegak即所有的k集合，m代表是第几个注意力头部，Wm是对注意力施加在value后的结果进行线性变换从而得到不同头部的输出结果，W'm 用于将xk变换成value，Amqk代表归一化的注意力权重。

由此可知，在Transformer的多头注意力计算中，每个query都要与所有位置的key计算注意力权重，并且对应施加在所有的value上。

再来看看我们（哦，不是我们，与我无关，你们要吗？）的Deformable Attetion：

![img](https://pic2.zhimg.com/80/v2-4616ccfab75bb6a494a91f6a0c15be6d_1440w.png)

和Transformer的很像是不是？（老师我没有抄作业，别凶..）可以看到，这里多了pq和delta_pmqk。其中，前者代表zq的位置（理解成坐标即可），是2d向量，作者称其为参考点(reference points)；而后者是采样集合点相对于参考点的位置偏移（offsets）。

可以看到，每个query在每个头部中采样K个位置，只需和这些位置的特征交互（x(pq+delta_pmqk)代表基于采样点位置插值出来的value），并不需要像Transformer般一开始先从全局位置开始学习才能逐渐过渡到关注局部（&稀疏的）的、真正有意义的位置。

需要注意的是，如可变形卷积一样，位置偏移delta_pmqk是可学习的，由query经过全连接层得到。

并且，注意力权重也一样，直接由query经过全连接层得到（因此，在可变形注意力机制下，其实没有真正所谓的key来与query交互计算，为何可以这样做，后文CW会谈自己的看法）！

同时在K个采样点之间归一化，而非像Transformer般是由query与key交互计算得出的。OK，顺着来，看看可变形注意力是如何应用到多尺度特征上的，依旧是公式走起：

![img](https://pic4.zhimg.com/80/v2-66ff210b895af8eb14ed44cca60dfa73_1440w.png)

这个也和上面的非常像是不是！？（老师我真的没有抄作业啊..太难了~）相比于上面，这里多了{xl}Ll=1和phil。

另外pq头上多了个小尖角，代表归一化到[0,1]，而phil正是用于将归一化的坐标映射（re-scales）到各个特征层去，这样，每个参考点在所有特征层都会有一个对应的(归一化)坐标，从而方便计算在不同特征层进行采样的那些点的位置。

至于{xl}Ll=1嘛，当然就是代表多尺度特征咯，xl代表第l层的特征。

在这里，每个query在每个特征层都会采样K个点，共有L层特征，从而在每个头部内共采样LK个点，注意力权重也是在这LK个点之间进行归一化。

另外，作者还提到，当L=K=1且且W'm 是identity矩阵时，该模块就退化成可变形卷积；相对地，当采样所有可能的位置（即全局位置）时，该模块等效于Transfomer中的注意力。

