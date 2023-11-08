---
layout: page
title: About
permalink: /about/
icon: heart
type: page
---

Notes, blogs on 

* `Programming`
* `Data analyzing`
* `Machine learning`
* `Linux` 
* and so on.

Drop me an email at [cyaeyz[at]gmail.com](mailto:cyaeyz@gmail.com) or leave a comment if you have any thoughts.

<!-- 
算法、数据分析及机器学习笔记。请使用 Mozilla Firefox、Google Chrome 等现代浏览器浏览本博客。

本博采用 Jekyll[[1]][1] 搭建，Markdown[[2]][2] 写作，托管于 GitHub[[3]][3]。 

 自 2016 年 07 月 07 日起，本站已运行 <span id="days"></span> 天，截至 {{ site.time | date: "%Y 年 %m 月 %d 日" }}，写了博文 {{ site.posts.size }} 篇，{% assign count = 0 %}{% for post in site.posts %}{% assign single_count = post.content | strip_html | strip_newlines | remove: ' ' | size %}{% assign count = count | plus: single_count %}{% endfor %}{% if count > 10000 %}{{ count | divided_by: 10000 }} 万 {{ count | modulo: 10000 }}{% else %}{{ count }}{% endif %} 字。 



若您觉得本博客所创造的内容对您有所帮助，可考虑略表心意，支持一下。

{% include reward.html %}

[1]: https://jekyllrb.com/ 'Jekyll'
[2]: http://daringfireball.net/projects/markdown/ 'Markdown'
[3]: https://github.com/ 'GitHub'
[4]: http://creativecommons.org/licenses/by-nc/3.0/cn/ '署名-非商业性使用 3.0 中国大陆' -->

{% include comments.html %}

<script>
var days = 0, daysMax = Math.floor((Date.now() / 1000 - {{ "2016-07-07" | date: "%s" }}) / (60 * 60 * 24));
(function daysCount(){
    if(days > daysMax){
        document.getElementById('days').innerHTML = daysMax;
        return;
    } else {
        document.getElementById('days').innerHTML = days;
        days += 10;
        setTimeout(daysCount, 1); 
    }
})();
</script>
