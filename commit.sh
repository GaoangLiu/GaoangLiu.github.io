#!/bin/bash 
# cd contents && bash update_post.sh 
comments=$@
git add .
git commit -m "$comments"
git push 

