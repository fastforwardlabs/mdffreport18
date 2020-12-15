## Experiments

While not comprehensive, this section seeks to provide some background on how different choices in the construction of these mappings affects the outcome of classification. All of our experiments can be found in a collection of Colab notebooks on our [github repo](https://github.com/fastforwardlabs/few-shot-text-classification).

### First, let's talk data

We explored text classification with two different datasets.

**AG News**

The original dataset consists of more than one million news articles gathered from more than 2000 news sources by ComeToMyHead (an academic news search engine) in more than one year of activity. We use a subset constructed for topic classification consisting of news articles in four categories. The training set consists of 30,000 articles for each category (120,000 total), while the test set contains 1,900 articles for each category (7,600 total). When performing “on-the-fly” classifications, we used only the test set. We pulled the AG News topic classification dataset from the open-source [Datasets](https://huggingface.co/docs/datasets/) [repository](https://huggingface.co/datasets) maintained by Hugging Face.

**[Reddit](https://www.aclweb.org/anthology/W17-4508/)**

This corpus contains nearly four million preprocessed submissions and comments from Reddit, collected between 2006 and 2016. It restricts to only those posts that include a “TL;DR” summary, which we use as the text we wish to classify into one of many possible subreddits (our target variable). We filtered this dataset down to the top 16 most popular subreddits by number of posts, resulting in over 600,000 examples. We randomly selected a stratified 10% of the posts as a test set, which we used in our “on-the-fly” classification. The full dataset can also be obtained from the Datasets repository maintained by Hugging Face.

### Improving on-the-fly classification by optimizing Zmap

Earlier, we talked about creating a mapping between SBERT and w2v spaces by comparing where individual words live in each of those spaces. This requires selecting a vocabulary of words. Which words create the best map? And how many words do you need?

#### How many words do you need?

Bigger isn’t always better. We explored the general effect on classification accuracy of a Zmap constructed from different numbers of words. We built a vocabulary from the most frequent words as measured over the corpus on which word2vec was trained. We first constructed Zmap from the top 100 most frequent words, then the top 200 most frequent words, and so on, until we’d included the top 100,000 most frequent words. With Zmap constructed, we then measured the accuracy of the resulting classification for both the AG News and Reddit datasets. The figure below shows the behavior we observed. As we increased the number of words used to construct Zmap, classification scores also increased, but only to a point. Eventually, after about 20,000 words, scores tended to decrease again. Why?

![Note: In order to focus on the general shape of these curves, we constructed this figure using relative accuracy (that is, the accuracy of each Zmap experiment is divided by the accuracy of the best experiment overall for that dataset). You can see how we constructed these curves in our Colab notebook.](figures/Zmaps_howmanywords.png)

The answer is likely largely due to the fact that we’re including rarer words, which could be interpreted as noise. The words are in descending order by frequency; this means the most common words are at the “top,” so as we include more words, we’re including rarer words.

The matrix, Zmap, is a fixed size because it maps from SBERT space (where embedding vectors are 768 elements long) to w2v space (where embedding vectors are only 300 elements long). This means that Zmap has dimension (768, 300), regardless of how many words are used to construct the mapping. When we have only a few words (say, 100), Zmap doesn’t contain enough information to provide a useful transformation, so the overall classification suffers. When we have 100,000 words, the rare words dilute the effect of the more common words and again, the overall classification suffers. 

There seems to be a sweet spot around 20,000 words—a finding we did not expect, but saw echoed in other datasets we tried (but do not show here). This peak is quite broad, and using anywhere from 15,000-25,000 words will likely optimize Zmap. However, using this optimized Zmap does not necessarily result in the best overall text classification. (We’ll explain this cryptic statement in the next section.)

### Which words make the best Zmap?

Instead of working in relative terms, let’s get specific, and explore the efficacy of Zmap. Below, we compare the classification accuracy of the latent text embeddings method using only SBERT representations, and transforming those representations with Zmap.

![]figures/accuracy_agnews_reddit.png

The teal bars represent classification accuracy when using only SBERT representations. We achieve over 50% accuracy for AG News, and nearly 40% accuracy for the Reddit dataset. While these might not seem like numbers to write home about, keep in mind that we did not perform any supervised training procedure! In general, scores are worse for the Reddit dataset because it’s simply harder to correctly classify into ten categories rather than only four. In both cases, scores reflect better-than-random accuracy, so we’re definitely on the right track. 

The orange bars illustrate the effect of transforming SBERT representations with a Zmap constructed from a vocabulary of 20,000 words. We observe a dramatic increase in performance for the AG News dataset: nearly 15 points! However, the news isn’t so rosy for the Reddit dataset, where we actually see a drop in classification accuracy. What gives?

To dig a bit deeper, let’s perform some cursory error analysis. Below, we break down the predictions for the AG News dataset by category for both the SBERT-only and SBERT+Zmap classification procedures. Each category contains 1,900 examples, so if our classification were perfect, each bar would fall exactly 1,900 examples in the diagram below. Looking at the teal bars (SBERT-only), we instead see a plurality of the news articles were predicted under the World category, with only a paltry few predicted as Sports. Once we transform these representations with Zmap (orange bars), we see the predictions begin to balance out between the four categories, which results in that dramatic 15 point accuracy increase.

![](figures/agnews_bycategory.png)

We speculate that the word “World” is so vague and all-encompassing that many news articles would end up being closer to this word than any other category name in SBERT space. Once we have a better mapping between SBERT space and w2v space (which is better at capturing the semantic meaning of individual words), more articles are predicted to be Sports. This is great! And it’s exactly what we expected Zmap to do. 

So why didn’t this work for the Reddit dataset? We constructed Zmap from a vocabulary of 20,000 most frequently used words in the corpus on which word2vec was originally trained, which happens to be a large portion of the Google News corpus. Perhaps this is why it performs so well with AG News, but not Reddit—words that are very common in one news corpus (Google News) might provide a better mapping for another news corpus (AG News) than for a collection of user posts and comments on Reddit.

To address this, we constructed a new Zmap from the 20,000 most frequently used words in the Reddit corpus. Below, we break down the predictions for the Reddit dataset as we did previously for the AG News dataset. This time, we show three bars for each category (subreddit name), corresponding to predictions using SBERT only, SBERT transformed with Zmap constructed from w2v words, and SBERT transformed with Zmap constructed from Reddit words. In this case, each category contains 1,300 examples, so if our classification were perfect all the bars would be the same height and fall exactly at 1,300 examples.

![](figures/reddit_bycategory.png)

Instead, we see something rather peculiar. Using SBERT representations alone (teal bars), we find posts are predicted under the “Funny” subreddit more than any other. “Funny” is a pretty general word, much like “World” was in the AG News example. However, this time, applying various Zmaps only serves to exacerbate the discrepancy! 

Rather than revealing a flaw in the method, we surmise that this is exactly what we should expect. While both “Funny” and “World” have broad, sweeping meanings, we argue that “Funny” is even more universal. Posts from any of these ten subreddits could easily be considered “Funny,” since humor is a common mode of communication among humans. Whether we’re talking about fitness or finance, we like to laugh!—and when we create better mappings between SBERT sentence space and w2v word space, we’re reinforcing that nearly everything can be funny. 

This example serves to highlight a limitation of the latent text embedding method: not only do category labels need to have semantic meaning, they also need to have specific semantic meaning to maximize the method’s utility. 

In general, an optimal Zmap will usually provide a better representation for classification. We find that constructing it from a vocabulary of 20,000 words is sufficient, and those words can come either from the w2v corpus, or from your own corpus. This technique is entirely unsupervised (as these words come either from open sources or from your own data), and it requires zero annotated examples.

So far, our experiments have focused on the performance of the latent text embedding method that does not rely on any labeled data whatsoever. Let’s now turn to another, perhaps more likely scenario: having some labeled data, but perhaps not enough to rely on traditional supervised classification techniques.

