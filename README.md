LDA: Latent Dirichlet Allocation 
---
This repository includes three open source versions of LDA with collapsed Gibbs Sampling, modified by nanjunxiao. 

[GibbsLDA++](http://sourceforge.net/projects/gibbslda/files/latest/download)  single thread,written in C++

[ompi-lda](http://code.google.com/p/ompi-lda/)  multi-node/multi-threads, written in C++

[online_twitter_lda](https://github.com/jhlau/online_twitter_lda)  multi-threads,written in Python

collapsed Gibbs LDA reference : [my blog](http://nanjunxiao.github.io/2015/08/07/Topic-Model-LDA%E7%90%86%E8%AE%BA%E7%AF%87/ )


What's New
---
#### 1. GibbsLDA++

fixed bugs:

1). memory leakage. 'delete[] p' instead of 'delete p',when p points to an Array. 

2). Array out of bound. (double)random() / RAND_MAX in [0,1]
```
int topic = (int)(((double)random() / RAND_MAX) * K);  -->  int topic = (int)(((double)random() / RAND_MAX + 1) * K);
double u = ((double)random() / RAND_MAX) * p[K - 1];   -->  double u = ((double)random() / RAND_MAX + 1) * p[K - 1];
```

#### 2. ompi-lda
fixed bug:

1). infer.cc bugs.

2). rm 'sampler.UpdateModel(corpus)' in lda.cc.

add features:

1). add theta twords file output.

2). add partial boost's hpp/cpp in include dir, so can make directly. 


#### 3. online_twitter_lda
add features:

1). add theta phi mat file output.


TODO
---
#### ompi-lda
1). twordsnum can configure.

2). rewrite cmd_flag without boost, so can remove include dir.

3). rewrite makefile.



 


