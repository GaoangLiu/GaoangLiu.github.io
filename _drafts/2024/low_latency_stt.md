---
layout:     post
title:      faster-whiper 打造语音聊天 bot
date:       2024-01-22
tags: [whisper,stt,asr]
categories: 
- nlp
---

如果使用 faster-whiper 结合 ChatGPT 打造一个语音聊天机器人。

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
  img {
    border-radius: 20px; /* 设置圆角的大小 */
  }
</style>


<div class="video-container">
  <iframe
  width="560" 
  height="315"
  src="https://www.youtube.com/embed/8FQJ8_A6O28"
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

<div class="video-container">
  <figure style="text-align: center;">
      <img src="https://image.ddot.cc/202401/stream-whisper-flow-v3.gif" width=678pt>
      <figcaption style="text-align:center"> faster-whisper 伪实时语音转写流程  </figcaption>
  </figure>
</div>



# 语音转写 
OpenAI 开源的 [whisper](https://github.com/openai/whisper)，最好的模型 `large v3` 英文字错率在 5% 左右，而中文的字错率约在 10% 左右，这个水平只能说勉强能用。实际使用体验，发音稍微不太清晰的情况下，就可能识别错误。发音相近的也容易转错，比如“三国演义”转写成“三观演义”。

faster-whiper 模型是在 whisper 的基础上进行了量化，速度上有 6 倍的提升。使用 kaggle 的 T4x2 GPU，实测一句 10 秒左右的语音，转写时间约在 0.7 秒以内，这个速度已经可以满足（伪）实时转写的需求了。

所上所述，音频是通过采样得到的，但音频的输入是连续的，whisper 只能离线转写，不支持流式转写。为了模拟流式转写，需要把音频切分成一小段，然后再转写。

音频的截取考虑了两种方案，一种是按固定时长截断音频输入，另一种是整句截取。前一种延迟有保证，但转写效果差；第二种转写效果好，但延迟不好控制。因为这个项目不是一个严格的实时转写，延迟 2 秒也好，5 秒也好，对体验影响不大，权衡转写效果与延迟，所以最终选择了第二种方案。

# 方案一：按时长截断音频输入 
使用 `pyaudio` 监听麦克风输入，每 30 毫秒采样一次，如果检测到音频超过一定时长，则截断音频，输出 wav 文件，然后调用 faster-whisper 进行转写。

这种方案的好处是实现简单，延时可以做到很低，比如每 0.5 秒截断并转写一次，那理论上延迟可以低到 0.5 秒。当然，前提是机器性能足够好，比如有一块 4090 GPU，此时转写的延迟可以忽略不计。

但缺点也很明显，就是转写错误比较多。而且截断的时候很有可能在一句话中间进行截断，导致转写时缺少上下文，转写错误率会更高。说到这点，刚好最近在 YouTube 上看到一个类似的 demo，视频主声称搞了一个超低延迟的语音转写服务，我分析了一下视频，发现有后期剪辑的痕迹，我猜测他可能是用了类似的方案，可以发现转写出来的结果中有明显的漏字现象。 

# 方案二：整句截取 
思路比较直观，借用 [`webrtcvad`](https://github.com/wiseman/py-webrtcvad) 监听麦克风输入，当检测到语音输入时，就开始录音，直到检测到有一段连续的静音时间，比如 0.5 秒，就认为一句话结束，然后输出 wav 文件，然后再调用 faster-whisper 进行转写。然后重复上述过程。

这种方案的好处是可以保证音频的完整性，拿到的音频基本是完成的一句话。 
但缺点是实现起来比较复杂，而且会带来一定的延迟，特别是在一句话比较长时，比如 10 秒以上，这种延迟就会比较明显。


<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/stream-whisper_20240129_1806.png" width=789pt>
    <figcaption style="text-align:center"> PyAudio + VAD + whisper 方案 </figcaption>
</figure>

实验过程中使用了 Kaggle 的 T4 GPU，然后按最长 0.5 秒的静音进行截断，本地录制完音频先同步到 `redislabs` 上的实例上，Kaggle 端的服务侧从 `redislabs` 上获取音频，然后进行转写。
实际用起来，多数情况下，延迟都能控制在 2 秒左右。但有时候为了迎合这个“静音检测”，说完一句话，需要刻意停顿一下。这个地方，也可能把静音检测的时长调短一点，比如 0.3 秒。 

当然，还有其他的优化方向，但都是一些算力换体验的方法。
如果 GPU 强劲，可以结合划窗录制音频的方案，录制完之后立即转写。监控到长时间的静音时，其实静音开始前的那一段已经转写完成了，直接返回这个结果即可。

最后，这个模拟实时转写的方案，是有极限的，无论是效果还是速度上，都不可能做到完美。但对于一些简单的场景，比如语音聊天 bot，这个方案已经足够了。 

# 语音聊天 bot
拿到文本之后，后面的就很简单了，调用聊天 bot 接口传入文本即可。Bot 的智能程度，取决于背后模型的能力，目前 GPT-4 还是当之无愧的 No.1，就是费用上有点高。

如果整个流程都要都使用开源免费模型，英文模型可以用 [mixtral 7x8b](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)，中文模型可以用零壹万物开源的 [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat)。 截止到 2024/01/29，Yi-34B-Chat 在 LMSYS Chatbot Arena Leaderboard 排行第 13，是中文模型中最好的。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/whisper-gpt_20240129_1754.png" width=789pt>
    <figcaption style="text-align:center"> Whisper + GPT 语音聊天 bot 方案</figcaption>
</figure>

# 代码与相关资源
- faster-whisper github 页面：https://github.com/SYSTRAN/faster-whisper
- openai/whisper: https://github.com/openai/whisper
- 代码仓库：https://github.com/ultrasev/stream-whisper


