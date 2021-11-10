---
layout:     post
title:      vim
date:       2021-11-10
tags: [vim]
categories: 
- editor
---

<img src="https://cdn.jsdelivr.net/gh/117v2/stuff@master/2021/2ecccf1e-9ef6-48cb-8568-c58b55bb2309.png" width=300pt>


# Change case in vim
## Change the case of a charactor

|Command|Function|
|:---|:----|
|`~`| change the case of current charactor|

## Change the case of a word

|Command|Function|Example|
|:---|:----|:----|
|`guw`| change to the end of current word from **upper to lower** | CH\#ANGE -> CH\#ange|
|`gUw`| change to the end of current word from **lower to upper** | ch\#ange -> ch\#ANGE|
|`guaw`| change all of current word from **upper to lower** | CH\#ANGE -> ch\#ange |
|`guiw`| change all of current word from **upper to lower** | CH\#ANGE -> ch\#ange |
|`gUaw`| change all of current word from **lower to upper** | ch\#ange -> CH\#ANGE |
|`gUiw`| change all of current word from **lower to upper** | ch\#ange -> CH\#ANGE |
|`g~w`| invert case to the end of current word | CH\#ANGE -> CH\#ange|
|`g~aw`| invert case of all of current word | CH\#ANGE -> ch\#ange|

Note: the \# symbol indicates the location of the cursor.

## Change the case of a line/sentence
|Command|Function|
|:---|:----|
|`guu`| change all words in current line from **upper to lower** |
|`gUU`| change all words in current line from **lower to upper** |
|`g~~`| invert case of all words in current line|
|`guG`| change to lowercase util the end of document|
|`gU)`| change to uppercase util the end of sentence|
|`gU}`| change to uppercase util the end of paragraph|
|`gU5j`| change 5 lines below to upper case|
|`gU5k`| change 5 lines above to upper case|