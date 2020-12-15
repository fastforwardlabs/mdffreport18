## Interpretability

Another benefit of the latent text embedding method is its inherent interpretability. Word2vec has been lauded for its interpretability, with the discovery that the numerical representations of words could be added and subtracted—the result of which would be the numerical representation of a word that completes an analogy. For example, the expression “Paris” + “France” - “Italy”  would yield an embedding vector that is very close to the word “Rome.” While it’s unclear if SBERT space operates similarly to w2v space (and what does it even mean to add and subtract documents?), having all the documents and labels embedded into the same latent space provides us with insight into why documents are predicted to have a given label. 

In the figure below, we embedded the AG News test set and the four category names (using SBERT*Zmap) and used the UMAP algorithm to render a two-dimensional embedding that we could visualize. We see that, in general, most articles cluster well around or near their label. The star represents a specific article that was predicted to have the label _Business_, and we can see that it is, in fact, closest to the label _Business_. This is the inherent power of latent embedding spaces for interpretability.

![](figures/agnews_umap.png)
