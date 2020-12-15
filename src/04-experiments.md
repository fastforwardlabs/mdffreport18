## Experiments

While not comprehensive, this section seeks to provide some background on how different choices in the construction of these mappings affects the outcome of classification. All of our experiments can be found in a collection of Colab notebooks on our [github repo](https://github.com/fastforwardlabs/few-shot-text-classification).

### First, let's talk data

We explored text classification with two different datasets.

**AG News**
The original dataset consists of more than one million news articles gathered from more than 2000 news sources by ComeToMyHead (an academic news search engine) in more than one year of activity. We use a subset constructed for topic classification consisting of news articles in four categories. The training set consists of 30,000 articles for each category (120,000 total), while the test set contains 1,900 articles for each category (7,600 total). When performing “on-the-fly” classifications, we used only the test set. We pulled the AG News topic classification dataset from the open-source Datasets repository maintained by Hugging Face.

**Reddit**
This corpus contains nearly four million preprocessed submissions and comments from Reddit, collected between 2006 and 2016. It restricts to only those posts that include a “TL;DR” summary, which we use as the text we wish to classify into one of many possible subreddits (our target variable). We filtered this dataset down to the top 16 most popular subreddits by number of posts, resulting in over 600,000 examples. We randomly selected a stratified 10% of the posts as a test set, which we used in our “on-the-fly” classification. The full dataset can also be obtained from the Datasets repository maintained by Hugging Face.
