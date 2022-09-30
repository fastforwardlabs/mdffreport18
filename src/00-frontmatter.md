# Few-Shot Text Classification

FF18 &middot; ©2020 Cloudera, Inc. All rights reserved

![Few-Shot Text Classification report cover](figures/ff18-cover-splash.png)

_This is an applied research report by <a href="https://www.cloudera.com/products/fast-forward-labs-research.html">Cloudera Fast Forward</a>.
We write reports about emerging technologies.
Accompanying each report are working prototypes or code that exhibits the capabilities of the algorithm and offer detailed technical advice on its practical application.
Read our full report on few-shot text classification below or <a href="/FF18-Few_Shot_Text_Classification-Cloudera_Fast_Forward.pdf" target="_blank" id="report-pdf-download">download the PDF</a>._ You can view and download the code for the accompanying text classification experiments <a href="https://github.com/fastforwardlabs/few-shot-text-classification" target="_blank">on Github</a>.

[[TOC]]

## Introduction
Text classification is a ubiquitous capability with a wealth of use cases, including sentiment analysis, topic assignment, document identification, article recommendation, and more. While [dozens of techniques](https://arxiv.org/abs/2004.03705) now exist for this fundamental task, many of them require massive amounts of labeled data in order to prove useful. Collecting annotations for your use case, however, is typically one of the most costly parts of any machine learning application, and demand continues to grow for techniques that make the most of small amounts of data. 

At Cloudera Fast Forward, we are no strangers to learning with limited data. We’ve covered multiple techniques in our research, including: [active learning](https://blog.fastforwardlabs.com/2019/04/02/a-guide-to-learning-with-limited-labeled-data.html), in which humans and machines collaborate to label data in a clever way, essentially bootstrapping from a small amount of labeled data; and [meta-learning](https://meta-learning.fastforwardlabs.com/), in which deep learning algorithms learn to learn.

There are several paradigms for learning from limited amounts of training data. Each of the scenarios outlined below can all be found in machine learning literature—and while definitions continue to evolve, these terms do still tend to have specific meanings.  

**Few-shot learning** for classification is a scenario in which there is a small amount of labeled data for all labels the model is expected to recognize. The goal is for the model to generalize to new unseen examples in the same categories both quickly and effectively.

In traditional **zero-shot learning**, a classifier is trained on one set of labels and evaluated on a different set of labels that it has never seen before (thus, “zero-shot”). This typically requires providing the model with some type of metadata about the unseen categories in order for it to generalize. (This is the scenario addressed by our [meta-learning report](https://meta-learning.fastforwardlabs.com/), in the context of image classification.) A variation on zero-shot learning is for the model to be evaluated both on never-before-seen and previously-seen labels; this is a more challenging task, because the model must recognize the difference.

Then there’s what we’ll call **“on-the-fly” learning**. This term is found in ML literature less often. We’ll use it in the context of classification with *no training examples at all*, and an undetermined number of labels. The goal of the model is to assign examples into some unknown set of possible categories on the fly. This type of model must leverage intrinsic information contained within the labels themselves.  

These learning paradigms are agnostic to the types of problems they attempt to solve, and can be found everywhere from computer vision to reinforcement learning. In this report, we’ll focus on text classification, and consider a classic method that can perform under any of these circumstances. Specifically, we’ll concentrate on text embeddings with a modern twist, and demonstrate their versatility as well as their limitations. Finally, we’ll provide insight into best practices for implementing this method. 
