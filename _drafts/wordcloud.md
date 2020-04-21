---
layout:     post
title:      Creating word cloud with Python
date:       2020-04-21
tags: [python, wordcloud]
categories: 
- Python
---

Word Cloud is a data visualization technique to represent text data by the frequency or importance of each word.
Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites. 

To generate a word cloud, the following modules are required:
```python
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd # pandas is optional 
```

# Producing wordcloud for movie subtitles
We are going to visualize the film [Gentlemen](https://movie.douban.com/subject/30211998/) by investigating its subtitle, which can be found [here](https://www.yts-subtitle.com/subtitles/thegentlemen20191080pwebripdd51x264-cm-15925).

<img class='center' src="https://i.loli.net/2020/04/21/Q3YvVqlH8zKR4xD.jpg"  alt="Gentlemen poster" width="320">

## Removing stop words
Step 1: we need to remove `stopwords`: words that are commonly used but contain few information, such as `a`, `an`, `the`. The `wordcloud` module provides a list of stop words (counts: 192) in [`wordcloud.STOPWORDS`](https://raw.githubusercontent.com/amueller/word_cloud/master/wordcloud/stopwords). 
Another stop word option is `nltk.corpus.stopwords.words('english')`, which contains 179 words. 

These two lists work fine in most cases, but not so good for texts built solely from conversations, such as subtitles from a film where dialogs contain lots of words that are not closely related to the film theme or topics. For example, we find that `know` and `now` occur frequently in the dialogs, but none of them carry much information about what is happening in that film. 

Thus, we've crafted our own stop word list. [This list](https://raw.githubusercontent.com/117ami/117ami.github.io/master/materials/stopwords.txt) contains 1527 words that are frequently adopted in daily conversations, but cause very few information loss when removed.

Now we can insert the following lines into our code whenever a longer list of stop words is necessary.

```python
import requests
stopwords = requests.get( \
    'https://raw.githubusercontent.com/117ami/117ami.github.io/master/materials/stopwords.txt'\
        ).text.split('\n')
```

## Generate word cloud
```python
title_file = 'gentlemen.txt'
wordcloud = WordCloud(width = 1600, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10
                ).generate(' '.join(open(title_file, 'r').readlines()))
  
# plot the WordCloud image                        
plt.figure(figsize = (16, 12), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 
```

And here comes the word cloud of **Gentlemen**, or `fuckman` if we combine the top two most significant word shown in the image.

<img class='center' src="https://i.loli.net/2020/04/21/JvGYdQuTXNhAnp5.png"  alt="word cloud of Gentlemen" width="800">


# References 
* [Generating word cloud in Python  - Geeks for geeks](https://www.geeksforgeeks.org/generating-word-cloud-python/)
* [A longer list of stopwords - Github repo](https://raw.githubusercontent.com/117ami/117ami.github.io/master/materials/stopwords.txt)

