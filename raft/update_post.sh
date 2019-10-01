#!/user/bin/bash 
# read draft folder, copy and rename blog files to _post
# rm _posts/*

post_drafts(){
	rm _posts/2019*
	files=$(ls drafts|grep md)
	for f in $files; do 
		date=$(cat drafts/$f | grep date | head -n 1 | awk '{print $2}')
		title=$(cat drafts/$f | grep title | head -n 1 | awk '{for(i=2;i<NF;++i)printf $i FS; printf $NF}')
		title="$date-${title// /-}.md"
		
		printf '%-30s %-30s\n' $f $title;
		rsync -vPrzq drafts/$f _posts/$title
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
|---|---|" | tee drafts/index.md 
	# prefix="http://www.fayderlau.xyz:404/blogs"
	prefix="{{site.baseurl}}"
	files=$(ls -t drafts|grep md)
	for f in $files; do 
		title=$(cat drafts/$f | grep title | head -n 1 | awk '{for(i=2;i<NF;++i)printf $i FS; printf $NF}')
		tight_title="${title// /-}"
		date=$(cat drafts/$f | grep date | head -n 1 | awk '{print $2}')
		echo "|[$title]($prefix/$tight_title) | $date |" | tee -a drafts/index.md
	done
}

update_index
post_drafts
