#!/user/bin/bash 
# read draft folder, copy and rename blog files to _post
# rm _posts/*
files=$(ls drafts|grep md)
for f in $files; do 
	date=$(cat drafts/$f | grep date | head -n 1 | awk '{print $2}')
	title=$(cat drafts/$f | grep title | head -n 1 | awk '{for(i=2;i<NF;++i)printf $i FS; printf $NF}')
	title="$date-${title// /-}.md"

	printf '%-30s %-30s\n' $f $title;
	cp drafts/$f _posts/$title
done
