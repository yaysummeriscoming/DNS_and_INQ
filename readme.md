This project contains Keras implementations of the Dynamic Network Surgery and Incremental Network Quantisation papers:

[Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493)


[Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044)

Code supports the Tensorflow and Theano backends.

Regarding Dynamic Network Surgery, the paper wasn't too clear about weight significance.  I had a poke around the author's C++ code and have used their implementation.  Basically they judge weights more than X standard deviations away from the mean as significant (denoted 'crate' in my and their code).  This seems to work quite well, as layers where all weights are fairly evenly distributed not too many are cut, whilst uneven distributions will result in a larger number of weights being cut.  You can see from the statistics printed to console that upper layers have more weights cut than lower layers.  This makes sense, as upper layers are more specialised in detecting specific patterns than lower layers, which function more like Gabor filters.

It's possible to combine these two techniques and include quantisation, but unfortunately I'm unable to share this implementation at this time.
