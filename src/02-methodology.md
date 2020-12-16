## Latent Text Embeddings

The NLP research team at Hugging Face recently published a [blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html) that detailed a handful of promising zero-shot text classification methods. We decided to dig a little deeper into one of these methods ourselves. It’s an oldie, but a goodie; we’ll explore how text embeddings can be used for classification.

First, an embedding method is used to generate a document representation and, separately, representations for each of the possible class labels. A document is then assigned the label that lies closest to it in the text embedding space. Note that we do not necessarily need the documents to be labeled a priori (in contrast to supervised learning, in which the model learns relationships explicitly from labeled examples). 

This method hinges on the idea that people can categorize documents into named categories without training, because we understand the meaning of the category labels. For example, when reviewing news articles, we can determine whether an article belongs under Science and Technology or Business (or both!) because the words “Science,” “Technology,” and “Business” each have semantic meaning associated with them. The intrinsic meaning of words (in particular, the class labels) is information that we leverage for classification. 

![](figures/ff18-01.png)

Let’s look at an example of how text embeddings can mimic this approach. First, suppose we have a collection of news articles that we’d like to classify into one of the following categories: *World News*, *Business*, *Science & Technology*, or *Sports*. Next, let’s assume we have a method (the “Embedding Model”) that can assign numeric vectors to text segments. We’ll use the Embedding Model to embed our news article and each of our labels into **latent space**. (A latent space is simply a compressed representation of the data in which similar data points are closer together).

Suppose this is one of our news articles: “[Breaking baseball barriers: Marlins announce first female GM in MLB history](https://www.kare11.com/article/sports/breaking-baseball-barriers-marlins-announce-first-female-gm-in-mlb-history/89-db8297a4-af08-4899-9e1f-38f9ddb3bcce).” We’ll pass the text of this article (along with the labels) through our Embedding Model, similar to (though not precisely) as shown below. 

![A news article and each of the labels are passed through the Embedding Model to generate vector representations for each text segment.](figures/ff18-02.png)

This produces embedding vectors which we can plot in our latent space: 

![Each of the vectors (one for the news article, and one each for the labels) can be represented in latent space. The article is closest to the word "Sports" in latent space, so *Sports* is assigned as the label.](figures/ff18-03.png)

We can now use a similarity metric (like cosine similarity) to compute which of the labels is closest to our news article in latent space, indicating that these text segments are the most similar. In this example, our article is closest to the word “Sports,” so we assign Sports as the label. This is because the word “Sports” is semantically similar to the word “Baseball,” which is the topic of our news article. It was this similarity between words and sentences that allowed us to label the news article—we didn’t use training data at all! 

Now, of course, we’ve left a lot out of the discussion. Most pressing: what is this mysterious Embedding Model? And what if we have some labeled examples? And is it really this simple? (Spoiler alert: not quite.) Let’s explore these questions. 

### The Embedding Model

The latent text embedding method of classification can be implemented with just about any type of text embedding model (and has been—see [these](https://www.aaai.org/Papers/AAAI/2008/AAAI08-132.pdf) [examples](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2016-174.pdf)), though some are definitely better than others! Which embedding model should we choose? In this section, we’ll review some popular approaches, though these are by no means exhaustive. 

Representing text numerically is not a new idea, and can be as simple as a **bag-of-words** or **tf-idf** vector, or as sophisticated as **[word2vec](https://arxiv.org/pdf/1301.3781.pdf)** or **[GloVe](https://nlp.stanford.edu/projects/glove/)**. Better performance is often achieved from contextual embeddings, like **[ELMo](https://arxiv.org/pdf/1802.05365.pdf)** or **[BERT](https://arxiv.org/abs/1810.04805)**, which embed words differently, depending on the other words in the sentence. These approaches focus on words, n-grams, or pieces of words as the basic embedding unit. Deriving a document representation then requires clever aggregation of the word embeddings.

In contrast, sentence embedding methods embed whole sentences or paragraphs; an early example is **[Doc2Vec](https://arxiv.org/pdf/1405.4053.pdf)**, which is similar to word2vec, but additionally learns a vector for the whole paragraph. More recent models include **[InferSent](https://arxiv.org/pdf/1705.02364.pdf)** and **[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)**.  Sentence embedding methods obviate the need for ad hoc aggregation techniques, and typically better capture the semantic meaning of whole text segments (as compared to aggregating word embeddings). 

Recent advances in sentence embedding methods have prompted us to re-evaluate the latent embedding approach. But first, what about BERT? Isn’t that all the rage these days for *everything* NLP-related? Shouldn’t we just use BERT to embed our text?

#### To BERT or not to BERT

Since its inception in 2017, BERT has been a popular embedding model.  As we’ll see, however, traditional ways of using BERT for semantic similarity are not ideal for a latent text embedding approach. 

BERT outputs an embedding vector for each input token, including word tokens and special tokens, such as SEP (a token that designates “separation” between input texts) and CLS. CLS is shorthand for “classification,” and this token was intended as a way to generate a sentence-level representation for the input. However, [experiments](https://arxiv.org/pdf/1908.10084.pdf) have shown that using the CLS token as a sentence-level feature representation drastically underperforms aggregated GloVe embeddings in semantic similarity tests.

Another option is to pool together the individual embedding vectors for each word token; this is the BERT equivalent of pooling word2vec vectors. However, these embeddings are not optimized for similarity, nor can they be expected to capture the semantic meaning of full sentences or paragraphs. While there are plenty of use cases where these embeddings do prove useful, neither is ideal as a latent space for semantic sentence similarity.  

Rather than using BERT as an embedding model, we can instead train it to specifically learn semantic similarity between sentences (an example of a sentence-pair regression task): BERT is shown many pairs of sentences, and is tasked with producing a score capturing their similarity. This procedure produces a fantastic semantic similarity classifier, but it is not efficient. In this scheme, BERT can only compare two text segments at a time. If we want to find the closest pair among many, we’ll have to pass every possible sentence pair through the model. Since BERT isn’t known for its speed, this procedure could take a while! 

Sentence-BERT (**SBERT**) addresses these issues. First [published](https://arxiv.org/abs/1908.10084) in 2019, this version of BERT is specifically designed for tasks like semantic similarity search and clustering—tasks that typically rely on cosine similarity to find documents that are alike. SBERT adds a pooling operation to the output of BERT to derive fixed sentence embeddings, followed by fine-tuning with a triplet network, in order to produce embeddings with semantically meaningful relationships. The result is a model that takes in a list of sentences and outputs a list of semantically salient sentence embeddings, one for each input sentence. These embeddings can be passed directly to cosine similarity, and the closest pair can quickly be determined. The authors demonstrate that SBERT sentence embeddings outperform aggregated word2vec embeddings and aggregated BERT embeddings in similarity tasks. Here is a latent space that we can leverage! SBERT (and SRoBERTa) models are publicly available both on the [Sentence-BERT website](https://www.sbert.net/) and in the [Hugging Face Model Repository](https://huggingface.co/models).

### Improving this approach with Zmap

Let’s take stock of and outline the latent text embedding method. For a document, _d_ (e.g., a news article), we want to predict a label, _l_, from a set of possible labels. We apply SBERT to our document, _d_, and to each of the _l_ labels, treating each label as a “sentence.” We then compute the cosine similarity between the document embedding and each of the label embeddings. We assign the label that maximizes the cosine similarity with the document embedding, indicating that these embeddings are most similar in SBERT latent space. This process can be succinctly expressed as:

![](figures/cosinesim_basic.gif)

We then repeat this for every document in our collection, and voilà! 

This actually works relatively well, depending (of course) on the dataset, and the quality and number of labels. But Sentence-BERT has been optimized… well, for sentences! It’s reasonable to suspect that SBERT’s representations of single words or short phrases like “Business” or “Science & Technology” won’t be as semantically relevant as representations derived from a word-level method, like word2vec or GloVe. This means, for example, that the word2vec representation of “Business” could well have a more meaningful relationship with other words in the word2vec latent space than its SBERT representation in SBERT latent space.  

![Left: In w2v latent space there tends to exist structure between similar words. Here, "Game" is a singular event in "Sports," while "Company" is a singular entity conducting "Business." Right: SBERT space is unlikely to have a similar structure between individual words, making it challenging to rely on SBERT label representations alone for classification. ](figures/ff18-04.png)

In a perfect world, we’d use SBERT to embed our documents, and w2v or GloVe to embed our class labels. Unfortunately, these embedding spaces do not have any inherent relationship between them, so we would have no way to know which labels were closest to our document. We could learn a relationship between these two spaces, but in order to do that, we'd need some annotated data—which defeats the purpose of zero-shot learning! 

![Left: Ideally we’d map SBERT sentence representations to w2v word representations but that requires labeled data. Right: By mapping words in SBERT space to those same words in w2v space we can learn an approximate mapping between the two latent spaces.](figures/ff18-06.png)

Instead, we can generate an approximation, by learning a mapping between individual words in SBERT space to those same words in w2v space. We begin by selecting a large vocabulary of words (we’ll come back to this) and obtaining both SBERT and w2v representations for each one. Next, we’ll perform a least-squares linear regression with l2 regularization between the SBERT representations and the w2v representations.^[It turns out that the solution to ordinary least squares with l2 regularization can be written as a concise equation, so we do not need to perform gradient descent to learn the weights. Instead, we need only invert a matrix. For intuition on how to interpret least squares as a linear algebra problem, check out this fantastic [blog post](https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b).]

This results in a matrix, _Z_, which maps SBERT space to w2v space. We’ll use _Z_ to transform both SBERT document representations (e.g., sentences) and SBERT label representations (e.g., words) into a new, lower-dimensional latent space, and perform our cosine similarity classification procedure in this new space.

![Once we've created a mapping, *Z*, between SBERT space and w2v space, we can use it to transform SBERT representations of sentences (such as news articles) and words (such as label names) into w2v space, incorporating the best of *both* worlds!](figures/ff18-07.png)

This is how our classification model looks now: 

![](figures/cosinesim_zmap.gif)

All we’ve done is to multiply _Z_ to both the document representation and the label representations, and then maximize the cosine similarity over the label set, as before. 

So where does this “large vocabulary of words” come from? One approach (used by the Hugging Face team) is to leverage the fact that w2v (and other popular open-source word embeddings) is trained on a massive corpus of text.  Most publicly released word representations are ordered by word frequency, with the most common words at the “top” and rare words at the “bottom.” This means that you can quickly identify a large vocabulary of the most frequently used words _in that corpus_. 

The assumption here is that learning a mapping between SBERT and w2v for the most commonly used words (as measured over a massive corpus of text) will provide a good mapping between the two latent spaces. Another approach we tried is to use the most frequent words in the corpus you wish to classify. In either case, the greatest benefit to both these approaches is that neither requires any annotated data—but each comes with pros and cons. (We’ll discuss those and explore the performance results in the Experiments section below.)

### Incorporating labeled data

Everything we’ve discussed so far has been under the premise that we have no labeled data whatsoever (“on-the-fly” learning). But what if you have _some_? The latent embedding approach is highly adaptable and can be modified to work with annotated examples in each category (few-shot learning), or when we might only have annotated examples for a subset of categories we’re interested in (traditional zero-shot learning). How do we take advantage of these labeled examples? We’ll follow the approach that the Hugging Face team outlines in their blog post, while hopefully providing a bit more context.

This method involves learning another mapping, this time between the documents and their labels—but we need to be careful not to overfit to our few annotated examples. Our goal is to learn a transformation that will rely on the semantic richness of our representations so far (i.e., multiplying SBERT embeddings by _Z_), while still allowing us to incorporate information from the labeled examples.

One way to accomplish this is to modify the regularization term in the linear regression. Before looking at this modification, let’s take a closer look at the traditional objective function for least-squares with l2 regularization in order to get a feel for how this is accomplished.

Weights, _W_, are learned through minimizing the loss function, as expressed below:

![](figures/lossfunction1.gif)

The first term essentially tells _W_ how to match an input, _X_, to an output, _Y_. The second term effectively minimizes the norm of the weights. The result is a set of regularized weights that map _X_ to _Y_ (which was exactly what we wanted in the previous section). Now we’ll modify the regularization term:

![](figures/lossfunction2.gif)

The first term still tells _W_ how to map _X_ to _Y_, but in the second term, the elements of the weight matrix are now pushed towards the identity matrix.^[As the Hugging Face team points out, this is equivalent to Bayesian linear regression with a Gaussian prior on the weights, centered at the identity matrix. Our prior belief is that our embedding mechanism, SBERT(d)Z, produces good text representations, and we only update this belief (move away from the identity matrix) as we see more training examples.]

If we only have very few examples, _W_ will likely be quite close to the identity matrix. This means that when we apply _W_ to our representations, SBERT(d)ZW will be very close to SBERT(d)Z. This is exactly what we want: to rely strongly on our original representations in the face of few examples. If we have many examples to learn from, _W_ will be pushed further away from the identity matrix, in which case it will more strongly modify the composition of SBERT(d)ZW, potentially changing the predicted label for the document, _d_. Our final classification procedure now looks like this:

![](figures/cosinesim_wmap.gif)

It’s important to note that this technique is now akin to supervised learning: _W_ is learned from training examples, and applied to test examples. However, notice that we have not specified whether _W_ is learned in a few-shot way (annotated examples for each relevant label) or in a zero-shot way (annotated examples for only a subset of the labels we are interested in). The approach is the same regardless, which is what makes this technique so flexible. 

So how well does it work? Let’s find out.
