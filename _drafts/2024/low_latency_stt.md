---
layout:     draft
title:      faster-whiper 打造语音聊天 bot
date:       2024-01-22
tags: [whisper,stt,asr]
categories: 
- nlp
---

使用 Faster-whiper 结合 ChatGPT 完成了一个语音聊天 bot 的搭建，整个过程中遇到了一些问题，这里做一个总结。

<style>
  .video-container {
    display: flex;
    justify-content: center;
    padding: 20px 10px;
  }

  iframe {
    width: 560px;
    height: 315px;
  }
</style>


<div class="video-container">
  <iframe
  width="560" 
  height="315"
  src="https://www.youtube.com/embed/m2AsSoZ43Xs"
  title="YouTube video player" 
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
  allowfullscreen
  style="border-radius: 12px;"></iframe>
</div>

整体的思路是：
- `pyaudio` 监听麦克风输入，每 30 毫秒采样一次，然后把采样到的音频数据转成 wav 格式
- `faster-whisper` 对 wav 格式的音频进行转写，获取文本
- `ChatGPT` 根据文本生成回复

# 语音转写 
OpenAI 开源的 [whisper](https://github.com/openai/whisper)，最好的模型 `large v3` 英文字错率在 5% 左右，而中文的字错率约在 10% 左右，这个水平只能说勉强能用。 实际使用下来，发音稍微不太清晰的情况下，就可能识别错误。比如“三国演义”转写成“三观演义”。


faster-whiper 是在 whisper 的基础上进行了量化，速度上有 6 倍的提升。使用 kaggle 的 T4x2 GPU，实测一句 10 秒左右的语音，转写时间约在 0.7 秒以内，这个速度已经可以满足（伪）实时转写的需求了。

# 方案一：按时长截断音频输入 
音频输入使用的是 `pyaudio`，监听麦克风输入，每 30 毫秒采样一次，如果检测到音频长度超过一定时长，比如一秒，则截断音频，输出 wav 文件，然后调用 faster-whisper 进行转写。这种方案的好处是实现简单，但缺点也很明显，就是会有一定的延迟，而且截断的时候可能会截断到一个词的中间，导致转写错误。


但好在 ChatGPT 有较强的 error correction 能力，再结合上下文，还是能理解对话的。 




