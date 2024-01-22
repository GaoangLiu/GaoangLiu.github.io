---
layout: post
title: vim skills
date: 2021-11-10
tags: vim
categories: editor
author: GaoangLiu
---
* content
{:toc}



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

# Visual select

- `Vap`: select around the current paragraph
- `V35G`: select current line up to line 35


## Pattern search
For example, `v / f o o` will select from your current position to the next instance of "foo." If you actually wanted to expand to the next instance of "foo,", for example, just press `n` to expand selection to the next instance, and so on.


# Comment / Uncomment 
## To comment out a block 
1. press `esc` (to leave editing or other mode)
2. hit `ctrl+v` (visual block mode, not `v`)
3. use the `↑/↓` arrow keys to select lines you want
4. `shift+i` (capital I)
5. insert the text you want, e.g. `#`
6. press `esc` twice. 

## To uncomment block
1. press `esc` (to leave editing or other mode)
2. hit `ctrl+v` (visual block mode, not `v`)
3. use the `↑/↓` arrow keys to select lines you want
4. press `d` or `x` to delete characters

refer to [so:quick way to comment/uncomment](https://stackoverflow.com/questions/1676632/whats-a-quick-way-to-comment-uncomment-lines-in-vim) for more solutions.

# Search and Replace

|Command | Function |
|:---- |:-----|
|`:s/foo/bar/3` | Find first occurrence of 'foo' (in three lines), and replace it with 'bar'. |
|`:s/foo/bar/g3` | Find each occurrence of 'foo' (in three lines), and replace it with 'bar'. |
|`:s/foo/bar/g` | Find each occurrence of 'foo' (in the current line only), and replace it with 'bar'. |
|`:%s/foo/bar/g` | Find each occurrence of 'foo' (in all lines), and replace it with 'bar'.|
|`:%s/foo/bar/gc` | Find each occurrence of 'foo' (in all lines), and replace it with 'bar'.|
|`:%s/\<foo\>/bar/gc`| Change only whole words exactly matching 'foo' to 'bar'; ask for confirmation.|
|`:%s/foo/bar/gi` | Change each 'foo' (case insensitive due to the i flag) to 'bar'; ask for confirmation.|

Q: search and replace the first N occurrance of foo with bar ?
1. when N is small, use `:%s/foo/bar/gc` and confirm replacement for N times.
2. when N is large, use `:call feedkeys(repeat("y", N) . "q") | %s/foo/bar/gc` 

## Search Range

|Command | Function |
|:---- |:-----|
|`:5,12s/foo/bar/g` | Change each 'foo' to 'bar' for all lines from line 5 to line 12 (inclusive).|
|`:.,+2s/foo/bar/g` | Change each 'foo' to 'bar' for the current line (.) and the two next lines (+2).|
|`:.,$$s/foo/bar/g` | Change each 'foo' to 'bar' for all lines from the current line (.) to the last line ($$) inclusive.|
|`:g/^baz/s/foo/bar/g` | Change each 'foo' to 'bar' in each line starting with 'baz'. |

