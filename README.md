# Go Frendi's static blog

This is the source code of my static blog. The blog has several prerequisites:

* `bash/zsh`
* `git`
* `hugo`
* `anaconda`

## Clone repository

You only need to do this once

```
git clone --recursive git://github.com/goFrendiAsgard/static-blog.git
```

## Init Virtual Environment and Add Conda Forge Channel

You only need to do this once

```
conda config --add channels conda-forge
conda config --set channel_priority strict 
make init-venv
```

## Start Virtual Environment

Do this everytime you want to work with this repo.

```
conda activate static-blog
```

## Writing Articles

You can write article in `markdown` or `notebook` format.

### Writing in Markdown

To write article in markdown, create a `<article-name>.md` file in `content/posts, and start writing.

Every article in markdown should be started with

```yaml
---
title: "<title>"
date: 2019-07-31T07:41:41+07:00
categories:
    - <category-1>
    - <category-2>
tags:
    - <tag-1>
    - <tag-2>
---
```

### Writing in Notebook

To write article in notebook, perform `make start-jupyter`, create a notebook, and start writing.

Every article in notebook should be started with `Raw NBConvert` as follow

```yaml
---
title: "<title>"
date: 2019-07-31T07:48:41+07:00
categories:
- <category-1>
- <category-2>
tags:
- <tag-1>
- <tag-2>
---
```

## Checking Up

## Deploy

Do this only when you are ready to publish the changes

```
./deploy.sh "<commit message>"
```
