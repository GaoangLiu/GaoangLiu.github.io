---
layout:     post
title:      Noise 
date:       2022-10-10
tags: [deeplearning, noise]
categories: 
- deeplearning
---


# Why Gaussian Noise is usually used?
高斯噪音就是幅度服从高斯分布的信号，之前有大佬说：“高斯噪音是自然界最恶劣的噪音”。

对于这个论断，从信息论上讲，它的依据是高斯噪音引入了最大的不确定性。根据[最大熵定理](https://baike.baidu.com/item/%E6%9C%80%E5%A4%A7%E7%86%B5%E5%8E%9F%E7%90%86/9938383)：如果平均功率受限，那么当信源符合高斯分布时，信源的熵最大。熵刻画了信息的不确定性，熵越大，说明噪声的不确定性越大，对信源的干扰也大。即如果假定功率受限，高斯噪声是最干扰信息的噪声。

从概率论上讲，根据中心极限定理，大量相互独立的随机变量，其均值的分布以正态分布为极限。现实世界中的噪声可能是由很多个来源不同的小的随机噪声累积形成的，即 $S = \sum_i{r_i}$ ，其中 $r_i$ 为随机噪声。另一方面在具有相同方差的所有可能的概率分布中，高斯分布在实数上具有最大的不确定性，也即是高斯分布是对**模型加入的先验知识量最少**的分布。

# 添加 Gaussian Noise
## 向信号中添加 Gaussian Noise
步骤很简单，就是先生成一个和原始数据维度一样的高斯分布的随机数，然后加到原始数据上。一个简单粗暴的实现如下：

```python
import numpy as np
dim = 100
noise = np.random.normal(0, 1, dim)
new_data = source_data + noise
```

> 但也有人建议，把生成的随机数与原始数据相乘。

上面的实现存在一个问题，就是如何确定噪声的范围，即如何设定高斯噪音中的期望与方差？

一种方法是通过信噪比(Signal to Noise Ratio, [SNR](https://en.wikipedia.org/wiki/Signal-to-noise_ratio))来控制噪声的范围。信噪比，顾名思义，即是信号与噪声的比例，计量单位为 dB，计算方式是 $10\log(Ps/Pn)$，其中 $Ps, Pn$分别表示信号和噪声的有效功率。信噪比大于 1，说明信号比噪声大，信噪比小于 1，说明噪声比信号大。信噪比越大，说明信号越清晰，噪声越小。
从放大器（比如音箱）的角度考虑，设备的信噪比越高表明它产生的噪声越少。一般来说，信噪比越大，说明混在信号里的噪声越小，声音回放的音质量越高。

因此，在确定 SNR 后，我们可以通过下面的代码来生成噪声，然后把噪声加到原始数据上，即得到引入噪声的数据。这种数据可以用来增强机器学习模型的鲁棒性，深度学习中的[去噪自编码器]({{site.baseurl}}/2022/10/10/autoencoder/)使用的也是这种思路。 

这种噪声也称为[**加性高斯白噪声**](https://zh.wikipedia.org/wiki/%E5%8A%A0%E6%80%A7%E9%AB%98%E6%96%AF%E7%99%BD%E5%99%AA%E5%A3%B0)（Additive white Gaussian noise, AWGN），因为噪声是可以叠加的，服从高斯分布，且是白噪声。

```python
# Additive white gassusian noise
import numpy as np

def awgn(source: np.ndarray, seed: int = 0, snr: float = 70.0):
    """ snr = 10 * log10( xpower / npower )
    """
    random.seed(seed)
    snr = 10**(snr / 10.0)
    xpower = np.sum(source**2) / len(source)
    npower = xpower / snr
    noise = np.random.normal(scale=np.sqrt(npower), size=source.shape)
    return source + noise

if __name__ == '__main__':
    t = np.linspace(1, 100, 100)
    source = 10 * np.sin(t / (2 * np.pi))
    # four subplots
    f, axarr = plt.subplots(2, 2)
    f.suptitle('Additive white gassusian noise')
    for i, snr in enumerate([10, 20, 30, 70]):
        with_noise = awgn(source, snr=snr)
        axarr[i // 2, i % 2].plot(t, source, 'b', t, with_noise, 'r')
        axarr[i // 2, i % 2].set_title('snr = {}'.format(snr))
    plt.show()
```

下图是 SNR 分别为 10dB, 20dB, 30dB, 70dB 的原始数据-带噪声数据对比图像，从图可见，当 SNR=70dB 时，几乎看不出噪声的存在。但仔细观察，还是能看到有的地方与原始分布有出入。这也是为什么麦克风导购博文中，良心博主一般给的建议是低于 80dB 的都不要考虑。

> 注： 目前国际电工委员会对信噪比的最低要求有规定，规定其前置放大器需要大于等于 63dB，后级放大器大于等于 86dB，合并式放大器大于等于 63dB。合并式放大器信噪比的最佳值应大于 90dB，CD 机的信噪比可达 90dB 以上，高档的更可达 110dB 以上。

<img src="https://file.ddot.cc/imagehost/2022/awgn.png" width=678pt>


# 参考
- [为什么深度学习去噪都采用高斯白噪声?](https://www.zhihu.com/question/67938028)
- [Signal to noise ratio, wikipedia](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)
- [Adding noise to a signal in Python, stackoverflow](https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python)
- [不同种类的噪声, wikipedia](https://en.wikipedia.org/wiki/Noise_(signal_processing))

