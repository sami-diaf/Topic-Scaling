# Topic Scaling
Learning time-based topics by combining document scaling and supervised topic models.

We applied Wordfish to estimate document scores that later serve as dependent variable for supervised Latent Dirichlet Allocation.
The goal is to learn time-based topics without requiring a time frame to be set.

The number of topics could be increased to unfold potential hidden structures in the corpus.

Example of application: State Of The Union (SOTU) addresses (1853-2019).

dtm_sotu.py gives estimates a Dynamic Topic Model [DTM](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf) on the SOTU corpus. Two topics are learned for each time frame (decade) using [Gensim](https://radimrehurek.com/gensim/models/wrappers/dtmmodel.html)
