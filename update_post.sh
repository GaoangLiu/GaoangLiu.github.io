#!/user/bin/bash 
# read draft folder, copy and rename blog files to _post
# rm _posts/*

post_drafts(){
	rm _posts/2019*
	files=$(ls _drafts|grep md)
	for f in $files; do 
		date=$(cat _drafts/$f | grep date | head -n 1 | awk '{print $2}')
		title=$(cat _drafts/$f | grep title | head -n 1 | awk '{for(i=2;i<NF;++i)printf $i FS; printf $NF}')
		title="$date-${title// /-}.md"
		
		printf '%-30s %-30s\n' $f $title;
		rsync -vPrzq _drafts/$f _posts/$title
	done

	echo -e "Drafts lifted to posts."
}

update_index() {
	today=$(date +%Y-%m-%d)
	echo "---
layout:     post
title:      index
date:       $today
img: 
tags: [index, blog]
---

✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦⚉✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

|Title|Date|
|---|---|" | tee _drafts/index.md 
	# prefix="http://www.fayderlau.xyz:404/blogs"
	prefix="{{site.baseurl}}"
	files=$(ls -t _drafts|grep md)
	for f in $files; do 
		title=$(cat _drafts/$f | grep title | head -n 1 | awk '{for(i=2;i<NF;++i)printf $i FS; printf $NF}')
		tight_title="${title// /-}"
		date=$(cat _drafts/$f | grep date | head -n 1 | awk '{print $2}')
		echo "|[$title]($prefix/$tight_title) | $date |" | tee -a _drafts/index.md
	done
}

# update_index
post_drafts
