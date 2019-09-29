---
layout: post
title: "[翻译] 简单的PHP模板引擎"
date: 2019-08-27 21:55:00 +0800
updated: 2019-08-27 21:55:00 +0800
author: Runbing
tags:
- PHP模板引擎
- PHP
- Smarty
categories:
- 编程
description: PHP 是一种有点罕见的语言，因为它不需要额外修改或库的支持，就已经在标记中实现将文本模板化。通过简单的 PHP 类就可以实现表示逻辑和应用逻辑的分离。而不用使用像 Smarty 这种看起来简洁但实际上臃肿的模板引擎。
---

<!-- ### PHP for templating? -->

### PHP 相当于模板？

<!-- PHP is a bit of a rare language as it can already template into text in markup with zero modifications or libraries. It is probably one of the big contributing factors why PHP is one of the most popular languages on the web today. (Can't be the only factor, it didn't work for ColdFusion) Most other web languages have a one or more templating languages with a different syntax that need to be learned on top of the implementing language. PHP lowers the bar to entry by allowing you to put your PHP code right into your html. But as we all know, sometime in your PHP tour, you will realize the need to separate presentation logic and the application logic. Some developers go running to some other solution that provides a different syntax. I am a bit puzzled on why this seems to be common practice, PHP can provide the same features without throwing another template syntax on top of what PHP already does. You can still achieve the separation needed with a simple class (shown at the end of this article). -->

PHP 是一种有点罕见的语言，因为它不需要额外修改或库的支持，就已经在标记中实现将文本模板化。这很可能是如今 PHP 成为最流行的 WEB 编程语言一员的重要促成因素之一（不能是唯一因素，因为它对 [ColdFusion](https://en.wikipedia.org/wiki/Adobe_ColdFusion) 不起作用）。大多数其它 WEB 编程语言拥有一种或多种具有不同语法的模板语言，这种语法需要在实现语言基础上进行学习。PHP 降低了进入门槛，允许你将 PHP 代码直接放进 HTML。

但是众所周知，在你 PHP 之旅的某个时刻，你将会意识到需要分离表示逻辑和应用逻辑。有些开发者开始使用其它提供不同语法的解决方案。我有点儿困惑于为何这似乎成了常见做法，PHP 能够提供相同的功能，不用在 PHP 已经做到的基础上增添另一种模板语法。你仍能够通过简单的类实现分离需求（文章末尾会展示）。

<!-- I mainly see two solutions that people tend to use over PHP. One solution is to use a premade template engine like Smarty on top of PHP. The other is to use string replacement techniques on a template file. -->

我主要见到人们趋向于在 PHP 之上使用两种解决方案。一种解决方案是在 PHP 基础上使用预制模板引擎，如 Smarty。另一种是在模板文件中使用字符串替换技术。

<!-- ### String replacement... sucks -->

### 字符串替换……很差劲

<!-- If you wrote your own template engine most likely using a `str_replace` or perhaps `preg_replace` to implement the embedding of dynamic parts of the code. There are two problems with this: One, it's slow; Secondly it's difficult to implement all the features needed to provide a robust templating language. Things like formatting functions, control structures etc are a bit clumsy to add to a solution like this. The other option is to implement very simple variable replacement, and then doing your formatting functions, control structures, etc. in your controller and just assign the result to variable replacement, however, that is completely against the point of having a template engine. The separation of presentation logic and app logic gets pretty blurry when you do some of the presentation logic outside of the template. -->

如果你编写自己的模板引擎，最有可能使用 `srt_replace` 或 `preg_replace` 实现代码动态部分的嵌入。这样做有两个问题：一是它很慢；二是它难以实现提供健壮模板语言所需要的所有功能。类似格式化函数、控制结构等添加到这种方案中有点儿蹩脚。另一个选项是实现非常简单的变量替换，然后执行格式化函数、控制结构等。在你的控制器中，只是把结果分配给变量替换，但是，那完全违反了使用模板引擎的意义。当你在模板之外执行一些表示逻辑时，表示逻辑和应用逻辑的分离会变得非常模糊。

<!-- Did I mention it's a pretty slow solution? -->

我是不是提到过这是一个非常慢的解决方案？

<!-- ### Smarty and other template engines -->

### Smarty 及其它模板引擎

<!-- Smarty and similar template engines are pretty darn redundant. Here is an example of the workflow for Smarty: -->

Smarty 和类似的模板引擎非常多余的。这里有一个 Sarty 工作流的示例：


<!-- * Smarty language is parsed
* Compiled to PHP
* PHP code is cached
* PHP code is parsed
* PHP code is compiled to opcodes
* If you have a opcode cache, opcodes are cached
* opcodes are ran -->

1. Smarty 语言被解析
2. 被 PHP 编译
3. PHP 代码被换成
4. PHP 代码被解析
5. PHP 代码被编译到 Opcodes
6. 如果你有 Opcode 缓存，Opcode 被缓存
7. Opcode 运行

<!-- If you don't see the redundancy there, I'm not doing my job very well. The whole idea of Smarty is like having a car on top of a car and believing it improves your gas mileage. Most people complain that the Smarty syntax is better than PHP's for templating. Bull. There is nothing really gained in Smarty's syntax, it only looks more concise, but in reality there is not enough gains to support having the bloat on top of PHP. You save a couple of keystrokes, big deal. <kbd>{$var}</kbd> vs. <kbd>&lt;?=$var?&gt;</kbd>. That is micro-optimization if I ever saw it. PHP control structres and formatting are much more concise and cleaner looking than Smarty's. Smarty doesn't work with most IDE's, so with PHP you gain everything you get with your IDE (or editor), code completion, highlighting, syntax linting, and more! -->

如果你没有看出多余的地方，是我做的不是很好。Smarty 的整个概念就像在一辆车的基础上再加一辆车，并且相信它能提高你的汽油里程。很多人抱怨 Smarty 语法在模板方面比 PHP 更好。胡扯。Smarty 的语法没有任何真正增益，它只是看起来比较简洁，但是实际上从它身上得到的好处并不值得让 PHP 更加臃肿。你节省了几次按键的敲击，真了不起。`{$var}` 对比 `<?=$var?>`。如果我能感受到优化，那也是微不足道的。PHP 结构控制和格式化比 Smarty 的更加简洁和清晰，Smarty 在大多数 IDE 中不被支持，而使用 PHP 你可以获得所有 IDE（或编辑器）带给你的一切，如代码自动完成、高亮显示、语法验证等。

<!-- ### common (lame) excuses -->

### 常见（蹩脚）的借口

<!-- #### My designers don't know PHP -->

#### 我的设计师们不懂 PHP

<!-- They also don't know the templating language you pick for them. If they are going to learn something, have them just learn enough PHP to do their templating. The syntax of something like smarty isn't really easier to learn at all. -->

他们也不懂你为他们挑选的模板语言。如果他们打算学习一些东西的话，让他们学习足够的 PHP 再去设计他们的模板。像 Smarty 这样的语法根本不容易学习。

<!-- #### I can't use PHP! that's not separating the presentation logic! -->

#### 我不能用 PHP！因为不能分离表示逻辑！

<!-- Actually you can achieve clear separation. See the class code below. -->

实际上你能做到清晰的分离。请看下方的类代码。

<!-- #### PHP syntax sucks for Templating -->

#### PHP 用于模板的语法糟糕透了

<!-- No, it's just fine. Really. -->

不，它刚刚好，真的。

<!-- #### I don't trust my designers with PHP -->

#### 我不相信我的设计师们使用 PHP

<!-- You are using a templating engine to solve the wrong problem. Template engines are meant to achieve higher maintainability through separation of logic. What you are trying to solve is a flaw in your job culture. These days, designers don't need to write bad server side code to cause a lot of hurt, they can write some horrid client side code that can be just as bad. Also your templating engine should be flexible in case you run into a wall in implementation. You don't want to paint yourself in a architectural corner because you don't do code review. (if you really don't trust your designers, only let them make static html mockups, have a jr. developer make them into templates.) -->

你正在用模板引擎解决错误的问题。模板引擎意味着通过逻辑的分离实现更高级可维护性。你所尝试解决的是你工作文化中的缺陷。如今，设计师们不需要编写糟糕的服务端代码去导致很多小问题，他们能够编写一些同样糟糕可怕的客户端代码。此外，你的模板引擎应该是灵活的，免得你在实现中碰壁。你不希望，因为你没有做代码审查（如果你真的不相信你的设计师们，只需要让他们制作一些静态 HTML 样板，让一名处级开发者将它们变成模板）。

<!-- #### I don't like PHP -->

#### 我不喜欢 PHP

<!-- Me either, that's why I work with Java and JSP. -->

我也是，这是为什么我使用 Java 和 JSP。

<!-- ## The Code: -->

## 代码

{% highlight php %}
<?php
class Template {
 private $vars = array();

 public function __get($name) {
  return $this->vars[$name];
 }

 public function __set($name, $value) {
  $this->vars[$name] = $value;
 }

 public function render($view_template_file) {
  extract($this->vars, EXTR_SKIP);
  ob_start();
  include($view_template_file);
  return ob_get_clean();
 }
}
{% endhighlight %}

<!-- Usage: -->

用法：

<!-- main.php template: -->

main.php 模板：

{% highlight html+php %}
<html>
<head>
  <title><?php echo $title; ?></title>
</head>
<body>
  <h1><?php echo $title; ?></h1>
  <div>
    <?php echo $content; ?>
  </div>
</body>
</html>
{% endhighlight %}

<p>content.php:</p>

{% highlight html+php %}
<ul>
  <?php foreach($links as $link): ?>
    <li><?php echo $link; ?></li>
  <?php endforeach; ?>
</ul>
<div>
  <?php echo $body; ?>
</div>
{% endhighlight %}

<p>controller.php:</p>

{% highlight php startinline %}
$view = new Template();

$view->title = "hello, world";
$view->links = array("one", "two", "three");
$view->body = "Hi, sup";

$view->content = $view->render('content.php');
echo $view->render('main.php');
{% endhighlight %}

<!-- ### Other PHP templating solutions you may want to check out -->

### 你可能想要查看的其它 PHP 模板解决方案

* [PHP Savant](http://phpsavant.com/)
* [Zend View, Part of the Zend Framework](http://framework.zend.com/manual/en/zend.view.html)

<!-- Added checks to make sure that no one tries to binding a variable named view_template_file to prevent someone doing something silly and bind the whole request to the function overriding the variable. Causing a nasty vulnerability。 -->

*编辑* 添加了检查以确保没有人尝试绑定名为 `view_template_file` 的变量去防止某人做某些蠢事，并将整个请求绑定到覆盖该变量的函数。导致严重的漏洞。

