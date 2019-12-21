[//]: # (author: samtenka)
[//]: # (change: 2019-12-19)
[//]: # (create: 2019-12-19)
[//]: # (descrp: )
[//]: # (to use: Open and read this document in a web browser.)

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Cicada: a Time-Dependent Bug Predictor

We model the bug pattern in a parameterized code base over time.  We imagine a
set of source files that is indexed by a fixed set but whose elements' contents
change over time.  Several categorical attributes determine the interaction of
modules and hence of code base behavior.  We seek to predict from data which 
files will be buggy given some a configuration of attributes. 

Many complications come to mind.  For example, one could statically analyze the
source ranging from shallow (dependency graphs) to deep (formal verification).
And one could learn the reliability of individual developers or of individual
code base versions.  But we will take a much simpler approach: we will just do
logistic regression; if that doesn't suffice, we can elaborate our model.

## 0. Model

We fit a function
