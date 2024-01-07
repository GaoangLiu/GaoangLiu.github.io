---
layout: post
title: HMM Segment
date: 2021-07-10
tags: nlp hmm segment
categories: algorithm
author: gaonagliu
---
* content
{:toc}


文章结构
1. HMM 基本概念
2. HMM 与分词关系 



3. 如何用 HMM 分词

在 [自然语言处理 -分词初窥](https://blog.csdn.net/SLP_L/article/details/112763427?spm=1001.2014.3001.5501) 我们介绍了基于词典的（最大匹配）分词方法，这种方法依赖于现有的词典库，对于新词（也称未登录词，out of vocabulary, OOV），则无法准确的进行分词。针对 OVV 问题，本文着重阐述下如何利用 HMM 实现基于字的分词方法。

利用 HMM 模型进行分词，主要是将分词问题视为一个**序列标注（sequence labeling）问题**。基本的思想就是**根据观测值序列找到真正的隐藏状态值序列**。在中文分词中，一段文字的每个字符可以看作是一个观测值，而这个字符的词位置（`BEMS`）可以看作是隐藏的状态。使用 HMM 的分词，通过对切分语料库进行统计，可以得到模型中 5 个要素：起始概率矩阵，转移概率矩阵，发射概率矩阵，观察值集合，状态值集合。结合这些要素，分词问题最终转化成求解隐藏状态序列概率最大值的问题，求解这个问题的一个常用方法是 Viterbi 算法。


## 隐马尔可夫模型（Hidden Markov Model，HMM）
HMM 包含如下五元组：

1. 状态值集合 $\mathcal{S}=\{s_1, s_2, ..., s_n\}$，其中 $n$ 为可能的状态数；
2. 观测值集合 $\mathcal{O}=\{o_1, o_2, ...,o_m\} $，其中 $m$ 为可能的观测数；
3. 转移概率矩阵 $T=[t_{ij}]$ ，其中 $t_{ij}$ 表示从状态i转移到状态j的概率；
4. 发射概率矩阵（也称之为观测概率矩阵）$E=[e_{jk}]$，其中 $e_{jk}$ 表示在状态 j 的条件下生成观测状态 k 的概率；
5. 初始状态分布 $\pi$。

### HMM 的三个问题
- **概率计算问题**，给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = (o_1, ..., o_T)$，怎样计算在模型 $\lambda$ 下观测序列 $O$ 出现的概率 $P(O|\lambda)$。可使用 Forward-backward 算法求解。
- **学习问题**，已知观测序列 $O$，估计模型 $\lambda = (A, B, \pi)$ 的参数，使得在该模型下观测序列的概率 $P(O|\lambda)$ 尽可能大，即用极大似然估计的方法估计参数。
- **解码（decoding）问题**，已知观测序列 $O$ 和模型 $\lambda = (A, B, \pi)$，求对给定观测序列条件概率 $P(S|O)$ 最大的状态序列 $S = (s_1, ..., s_T)$，即给定观测序列，求最有可能的对应的状态序列。


## HMM 分词预备知识
在[之前的文章中](https://blog.csdn.net/SLP_L/article/details/112763427?spm=1001.2014.3001.5501)我们简要介绍了基于字分词的方法是如何**将分词转换为序列标注问题**。在 HMM 分词中，我们将状态值集合 $\mathcal{S}$ 置为 {'B', 'M', 'E', 'S'}，分别表示词的开始、结束、中间（begin、end、middle）及字符独立成词（single）；观测序列即为中文句子。比如，“今天天气不错” 通过 HMM 求解得到状态序列 "B E B E B E"，则分词结果为“今天/天气/不错”。

形式化的，分词任务对应的问题可表述为：对于观测序列$C=\{c_1,...,c_n\}$，求解最大条件概率: $P(S|C) = \text{argmax}_{s_1, ..., s_n}P(s_1, ..., s_n|c_1, ..., c_n)$，其中 $S$ 表示（隐藏）状态序列，$s_i$ 表示字符 $c_i$ 对应的状态。

由 bayes 公式，我们可以得到 $\text{argmax}P(s_1, ..., s_n|c_1, ..., c_n) = \text{argmax}P(c_1, ..., c_n|s_1, ..., s_n) \cdot P(s_1, ..., s_n)$。 
参照关于 n-gram 分词的做法，我们做以下两点假设来减少稀疏问题:
1. 齐次马尔可夫性假设，即假设隐藏的马尔科夫链在任意时刻 t 的状态只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻t无关：$P(s_i|s_{i-1}, ..., s_1) = P(s_i|s_{i-1})$。
2. 观察值独立性假设(观察值只取决于当前状态值)：$P(c_1, ..., c_n|s_1, ..., s_n) = P(c_1|s_1) \cdots P(c_n|s_n)$。

再结合以上 bayes 公式得到的结果，我们有 

$$P(s_1, ..., s_n|c_1, ..., c_n) = P(c_1|s_1) P(s_1) \cdot P(c_2|s_2) P(s_2|s_1) \cdots P(c_n|s_n) P(s_n|s_{n-1})$$

分词问题中状态 $\mathcal{S}$ 只有四种，即 {B,E,M,S}，其中 $P(\mathcal{S})$ 可以作为先验概率通过统计得到，而条件概率 $P(C|\mathcal{S})$ 即汉语中的某个字在某一状态的条件下出现的概率，也可以通过统计训练语料库中的频率来估算。

**注**： 
当观察序列比较长时，为避免计算值下溢，我们通过采用对概率值 $P(c_i|s_i), P(s_i|s_{i-1})$ 取对数，然后计算 $\log(P(c_1|s_1)) + \log(P(s_1)) + \cdots + \log(P(c_n|s_n)) + \log(P(s_n|s_{n-1}))$ 的和来找出数值最大的 $\log(P(S|C))$，即概率值最大的 $P(S|C)$。


### Q&A
Q: 为何不基于统计先验直接估算概率 $P(S|C)$ ?

A: 如果采用这种方法，我们需要估算多个状态的联合概率 $P(S)$ 及条件概率 $P(C|S)$，当观察序列比较长或者遇到之前没有出现过的序列，概率值 $P(C|S)$ 对于任意状态序列 $S$ 都为 0， 我们就无法估计最有可能的状态序列 $S$，进而无法获取有用的分词信息。



## HMM 分词实践
### Viterbi 算法
求解最优状态序列可以采用 [Viterbi算法](https://blog.csdn.net/SLP_L/article/details/115394766?spm=1001.2014.3001.5501)。Viterbi 算法本质上是一个动态规划算法，利用到了状态序列的最优路径满足这样一个特性：**最优路径的子路径也一定是最优的**。

### 示例与实现
分词使用的语料来自于 SIGHAN Bakeoff 2005 的 [icwb2-data.rar](http://sighan.cs.uchicago.edu/bakeoff2005/)，目录 `training` 内容如下图:
```bash
training/
├── as_training.b5
├── as_training.utf8
├── cityu_training.txt
├── cityu_training.utf8
├── msr_training.txt
├── msr_training.utf8
├── pku_training.txt
└── pku_training.utf8
```

我们使用 `training/msr_training.utf8` 进行分词预训练，需要估算以下几个概率：
1. 初始状态概率
2. 状态转移概率矩阵
3. 状态发射概率矩阵

#### 初始状态概率
<!-- 从 `msr_training.txt` 的语料中我们统计初始状态的出现的概率值， -->
我们设置词位 `B` 与 `S` 的初始概率均为 `-0.6931471805599453`(`math.log(0.5)`)，而 `M`, `E` 的概率为一个极小的值。
```python
MIN_FLOAT=-1e10
start_p = {'B': -0.6931471805599453, 'M': MIN_FLOAT, 'E': MIN_FLOAT, 'S': -0.6931471805599453}
```

`M, E` 的初始值设定为 `-1e10` 是因为开始词位为 `S` 或者 `B` 的语句更符合我们的语言习惯，例如平时我们会说: “我们去打球吧。”，但不会说“们去打球吧。”。加大两个词位的初始值是为了避免找出以他们为开始的词位序列。

#### 状态转移概率
也即是词位 {'B', 'M', 'E', 'S'} 之间相互转移的概率矩阵。我们从语料 `msr_training.txt` 中统计得到的概率矩阵（数值取对数）如下，其中 `-10000000000` 即上面已经提及到 `MIN_FLOAT` 数值，因为 `B->B`、`B->S`、`M->S`、`M->B`、`S->M`、`S->E`、`E->M`、`E->E` 为不可能事件，我们将其概率值设定一个较小的数值 `MIN_FLOAT`。
```python
trans_p = 
    [[-10000000000.0, -1.8272249217840997, -0.1753769420145616, -10000000000.0],  # B -> B, M, E, S
    [-10000000000.0, -0.7020775100490347, -0.6842958964748135, -10000000000.0],   # M -> B, M, E, S
    [-0.5077169517050144, -10000000000.0, -10000000000.0, -0.9209719351182197],   # S -> B, M, E, S
    [-0.45101686842012895, -10000000000.0, -10000000000.0, -1.0132976176788933]]  # E -> B, M, E, S
```

#### 状态发射矩阵
以下表中为我们估算出来的发射概率(数值取对数)部分结果。比如，`P('我', 'B')`表示状态为 `M` 的情况下出现“我”这个字的概率。
```python
emit_p = {
    ('我', 'S'): -5.162818030524261,
    ('我', 'B'): -5.2928634822609295,
    ('我', 'E'): -8.91047834678911,
    ('我', 'M'): -9.500310449945923,
    ...}
```

#### 实现效果
从下图分词效果可以看出，我们训练得到的 HMM 的效果只能算还可以，能准备切分诸如 `世博园`、`世博员` 并不受歧义词 `来世`、`去世` 的影响，但也有很多词切分的不准确，如红圈部分所示。 一部分是因为概率估算不够精准，另一部分是因为有新词出现，比如 `腾讯`。
<img src="https://i.loli.net/2021/04/09/8U4tRFpMinAWkwE.png" width='567px'>



#### 在测试集上评测
我们的模型是在 `msr_training_words.utf8` 上训练得到到，测试应该使用对应的测试集文件  `test/msr_testing.utf8`，将分词结果输出到文件 `result.txt` 之后，我们使用 `scripts` 下 Perl 测试脚本 `score` 进行测试，命令如下：
```bash
./scripts/score gold/msr_training_words.utf8 gold/msr_test_gold.utf8 result.txt > summary.txt
```

最终得到的结果如下:
1. 召回率 `0.741`
2. 准确率 `0.787`
3. F1 `0.763`

这个分词结果非常一般，作为对比，使用之前介绍过的[基于字典最大匹配](https://blog.csdn.net/SLP_L/article/details/112763427?spm=1001.2014.3001.5501)的方法，我们可以得到 `0.918` 的准确率及 `0.957` 的召回率。
对于当前的任务，HMM 分词效果较差的一个原因是其“自由度”太大，比如对于句子 "英国白金汉宫发表声明，菲利浦亲王去世。"，HMM 得到分词结果 `[英国 白金 汉宫 发表 声明 ， 菲利浦亲 王去 世]`，结果出现了字典中不存在的词 “菲利浦亲”、 “王去”。但自由度大也是 HMM 的优势之一，相对于基于字典的分词，HMM 能够更好的发现新词。 

```bash
TRUE WORDS RECALL:	0.867
TEST WORDS PRECISION:	0.867
=== SUMMARY:
=== TOTAL INSERTIONS:	10514
=== TOTAL DELETIONS:	3894
=== TOTAL SUBSTITUTIONS:	18900
=== TOTAL NCHANGE:	33308
=== TOTAL TRUE WORD COUNT:	106873
=== TOTAL TEST WORD COUNT:	113493
=== TOTAL TRUE WORDS RECALL:	0.787
=== TOTAL TEST WORDS PRECISION:	0.741
=== F MEASURE:	0.763
=== OOV Rate:	0.026
=== OOV Recall Rate:	0.253
=== IV Recall Rate:	0.801
###	result.txt	10514	3894	18900	33308	106873	113493	0.787	0.741	0.763	0.026	0.253	0.801
```


#### 完整代码 
```python
import math
import pickle
from collections import defaultdict
from typing import List, Tuple

import codefast as cf


class HMM:
    def __init__(self):
        cur_dir = io.dirname()
        self.TRAIN_CORPUS = f'{cur_dir}/data/msr_training.utf8'
        self.emit_pickle = f'{cur_dir}/data/emit_p.pickle'
        self.trans_pickle = f'{cur_dir}/data/trans.pickle'
        self.punctuations = set('，‘’“”。！？：（）、')
        self.MIN_FLOAT = -1e10
        self.emission = None
        self.transition = None

    def train(self) -> Tuple[dict, list]:
        ''' train HMM segment model from corpus '''
        if self.emission and self.transition:     # to avoid I/O
            return self.emission, self.transition

        if io.exists(self.emit_pickle) and io.exists(self.trans_pickle):
            emit_p = pickle.load(open(self.emit_pickle, 'rb'))
            trans = pickle.load(open(self.trans_pickle, 'rb'))
            self.emission, self.transition = emit_p, trans
            return emit_p, trans

        emit_p = defaultdict(int)
        ci = defaultdict(int)
        # b0 m1 e2 s3
        trans = [[0] * 4 for _ in range(4)]

        def update_trans(i, j):
            if i < 0:
                return j
            trans[i][j] += 1
            return j

        with open(self.TRAIN_CORPUS, 'r') as f:
            for ln in f.readlines():
                pre_symbol = -1
                _words = (e for e in ln.split(' ') if e)
                for cur in _words:
                    if cur in self.punctuations:
                        pre_symbol = -1
                    else:
                        if len(cur) == 1:
                            emit_p[(cur, 'S')] += 1
                            ci['S'] += 1
                            pre_symbol = update_trans(pre_symbol, 3)
                        else:
                            for i, c in enumerate(cur):
                                if i == 0:
                                    emit_p[(c, 'B')] += 1
                                    ci['B'] += 1
                                    pre_symbol = update_trans(pre_symbol, 0)
                                elif i == len(cur) - 1:
                                    emit_p[(c, 'E')] += 1
                                    ci['E'] += 1
                                    pre_symbol = update_trans(pre_symbol, 2)
                                else:
                                    ci['M'] += 1
                                    emit_p[(c, 'M')] += 1
                                    pre_symbol = update_trans(pre_symbol, 1)
        cf.info('count pairs complete.')

        for i, t in enumerate(trans):     # normalization
            trans[i] = [
                math.log(e / sum(t)) if e > 0 else self.MIN_FLOAT for e in t
            ]

        for key, v in emit_p.items():
            emit_p[key] = math.log(v / ci[key[1]])

        with open(self.emit_pickle, 'wb') as f:
            pickle.dump(emit_p, f)

        with open(self.trans_pickle, 'wb') as f:
            pickle.dump(trans, f)

        return emit_p, trans

    def calculate_trans(self, emit_p: dict, ci: dict, obs: str,
                        state: str) -> float:
        return (1 + emit_p.get((obs, state), 0)) / ci.get(state, 1)

    def viterbi(self, text: str, emit_p: dict, trans_p: dict) -> List[str]:
        ''' a DP method to locate the word segment scheme with maximum probability 
        for a given Chinese sentence.
        :param text: str, observed sequence, e.g., '人性的枷锁'
        :param emit_p: dict, emission probability matrix, e.g., emit_p[('一', 'B')] = 0.0233
        :param trans_p: dict, transition probability matrix, e.g., trans_p['B']['M'] = 0.123
        :return: list[str], word segments, e.g., ['人性', '的' ,'枷锁']
        '''
        if not text:
            return []
        state_index = dict(zip('BMES', range(4)))
        cache = {}

        for i, c in enumerate(text):
            if i == 0:
                for s in 'BS':
                    cache[s] = (-0.5, s)     # this initial prob is customizable
                cache['E'] = (self.MIN_FLOAT, 'E')
                cache['M'] = (self.MIN_FLOAT, 'M')
            else:
                cccopy = cache.copy()
                for s in 'BMES':
                    max_prob, prev_seq = float('-inf'), ''
                    for prev_state, v in cccopy.items():
                        prev_index, cur_index = state_index[
                            prev_state], state_index[s]

                        # not '*' but '+'
                        new_prob = v[0] + trans_p[prev_index][
                            cur_index] + emit_p.get((c, s), self.MIN_FLOAT)
                        if new_prob > max_prob:
                            max_prob = new_prob
                            prev_seq = v[1]
                    cache[s] = (max_prob, prev_seq + '->' + s)

        # assume a sentence ends with either 'E' or 'S'
        # print(cache)
        seq = cache['E'][1] if cache['E'][0] > cache['S'][0] else cache['S'][1]
        return self._cut(text, seq)

    def _cut(self, text: str, seq: str) -> List[str]:
        ''' seperate a sequence by word lexeme.
        e.g., input ('人性的枷锁', 'B->E->S->B->E'), output ['人性','的','枷锁']
        '''
        res = []
        # print(seq, text)
        # print(len(text))
        for a, b in zip(text, seq.split('->')):
            if (b == 'B' or b == 'S'):
                res.append('')
            res[-1] += a
            # print(res)

        return res

    def segment(self, text: str) -> List[str]:
        emit_p, trans = self.train()
        res, prev_text = [], ''
        for i, c in enumerate(text):
            if c in self.punctuations:
                res += self.viterbi(prev_text, emit_p, trans)
                res.append(c)
                prev_text = ''
            else:
                prev_text += c

        if prev_text:
            res += self.viterbi(prev_text, emit_p, trans)
        return res


if __name__ == '__main__':
    texts = [
        '看了你的信，我被你信中流露出的凄苦、迷惘以及热切求助的情绪触动了。', '这是一种基于统计的分词方案', '这位先生您手机欠费了',
        '还有没有更快的方法', '买水果然后来世博园最后去世博会', '欢迎新老师生前来就餐', '北京大学生前来应聘', '今天天气不错哦',
        '就问你服不服', '我们不只在和你们一家公司对接', '结婚的和尚未结婚的都沿海边去了', '这也许就是一代人的命运吧',
        '改判被告人死刑立即执行', '检察院鲍绍坤检察长', '腾讯和阿里都在新零售大举布局', '人性的枷锁'
    ]

    texts += [
        '好的，现在我们尝试一下带标点符号的分词效果。', '中华人民共和国不可分割，坚决捍卫我国领土。',
        '英国白金汉宫发表声明，菲利浦亲王去世，享年九十九岁。', '扬帆远东做与中国合作的先行', '不是说伊尔新一轮裁员了。',
        '滴滴遭调查后，投资人认为中国科技业将强化数据安全合规。'
    ]
    for text in texts:
        ret = ' '.join(HMM().segment(text))
        print(ret)
```


## reference
- https://www.cnblogs.com/zhbzz2007/p/6092313.html
- 《统计学习方法》，李航
- 《机器学习》，Tom M.Mitchell


