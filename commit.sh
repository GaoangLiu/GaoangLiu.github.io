#!/bin/bash 
comments=$1
echo '*' `date +"%a %D/%T":` $comments | tee -a log/c.log 
readme='log/log.md'

echo '
README
***** 
|Author|@ssrzz|
|:---  |:---
|E-mail|ssrzz@pm.me

### Log: 
```' > $readme

cat log/c.log >> $readme
echo '```' >> $readme 

git add .
git commit -m "$comments"
git push 