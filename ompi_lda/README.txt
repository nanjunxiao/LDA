
====================
INTRODUCTION
====================

I name this project by ompi-lda, which shorts for OpenMP and MPI based
parallel implementation of LDA (Latent Drichlet Allocation).

This package delivers the following four LDA training programs:

1. single thread on single node
2. multi-thread on single node
3. single thread on multiple nodes
4. multi-thread on multiple nodes  (multithreading + distributed computing)

The multi-threading functionality depends on OpenMP.  I tested this
functionality using GCC 4.2.1 on Mac OS X (Snow Leopard).

The multiple-node (distributed computing) functionality depends on
MPI.  I tested this functionality using MPICH2 on Snow Leopard, Ubuntu
and Cygwin (Windows XP).

====================
ALGORITHMS
====================

The parallel algorithms I implemented in this package can be found in
the following papers:

* David Newman, Padhraic Smyth, Mark Steyvers, Scalable Parallel Topic
  Models, Journal of Intelligence Community Research and Development
  (2006).

* David Newman, Arthur Asuncion, Padhraic Smyth, MaxWelling,
  Distributed Inference for Latent Dirichlet Allocation.  NIPS 2007.

====================
HISTORY
====================

This package enhances plda-3.0 (http://code.google.com/p/plda), which
was initialized by Hongjie Bai and I when I was working in Google, and
is under development by Hongjie after I left Google and my name was
removed from this OPEN SOURCE project by my former manager Edward
Chang.  plda implements the serial (standard) Gibbs sampling training
algorithm and an MPI-based parallel algorithm.  plda consists of
logically two parts of code:

1. The Gibbs sampling algoirthm and utilities, which was written by
   Matt Stanton and I for a Google MapReduce based LDA implemention.

2. MPI-based training algoirthm with a highly-efficient rewrite of LDA
   model data structure, which was committed by Hongjie Bai.

Our paper describing plda

* Yi Wang, Hongjie Bai, Matt Stanton, Wenyen Chen. Parallel Latent
  Dirichlet Allocation for Large-Scale Applications.  Algorithmic
  Aspects in Information and Management, 2009.

is still useful in explaining the MPI aspect of ompi-lda.

====================
WHY OPENMP
====================

Most computers (either works alone or in a computer-cluster) have CPUs
with multiple cores.  To better utilize such equipment, I decided to
make use of multi-threading.  Fortunately, Gibbs sampling algorithm
for LDA has a clear logic and can be parallelized easily using OpenMP
(without the need of more flexible but harder to use utilities like
pthread and boost thread library).  It is true that OpenMP requires
support from the C++ compiler, but most well-used compilers nowadays,
include G++, supports OpenMP well.  David Newman's paper provides a
clear guidence to this work.
