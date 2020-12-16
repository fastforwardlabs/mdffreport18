## Limitations

While the latent text embedding approach provides a flexible and semi-interpretable way to classify text, it’s not without its limitations. 

### Validation is a challenge

Let’s address the elephant in the room: we had access to labeled training and test sets for all the experiments performed for this report, which is how we were able to assess the performance of Zmap. In a real-world, “on-the-fly,” no-labeled-data-available situation, validating the results of the method is essentially impossible. This is why we spent time looking for possible generalities that could provide guidance on how to use this method in a practical application—but, as we saw with the Reddit dataset, sometimes Zmap actually makes your classification accuracy worse! And without labeled data to validate the method, you will have no way of knowing if this is the case for your data.  

This isn’t solely an issue with the latent text embedding approach—this is a challenge for any unsupervised learning situation. The solution, unfortunately, is to simply buckle down and label some data! As we saw, even just a couple hundred examples can provide a wealth of insight and performance gains.

### Meaningful labels are a necessity

It’s not enough to have a few labeled examples for training or validation; care must be taken when deciding what the label names themselves will be. This method relies on labels laden with meaning, and that possess some semantic relationship to the text documents you wish to classify. If, for example, your labels were _Label1_, _Label2_, and _Label3_, this method would not work, because those labels are meaningless.

In addition to being meaningful, label names should not be too general. As we saw, the words “World” and “Funny” were too broad and all-encompassing to be practical label names. While an optimal Zmap was able to correct for this effect in the _World_ example, this won’t always be the case, as we saw with the label _Funny_.

### This probably won’t beat supervised methods

Finally, in terms of performance metrics like accuracy or F1 score, the latent text embedding approach won’t beat out standard supervised text classification methods. We saw that even the best optimization of Wmap could only increase the classification accuracy by 10-15 points, and training on more labeled examples didn’t help. If you have a good amount of labeled data, it’s worth checking out traditional supervised approaches first.
