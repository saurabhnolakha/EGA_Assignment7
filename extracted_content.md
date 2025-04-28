Toggle the table of contents

# Word embedding

23 languages

* العربية
* Català
* Čeština
* Deutsch
* Español
* Euskara
* فارسی
* Français
* 한국어
* Italiano
* עברית
* 日本語
* Norsk bokmål
* Polski
* Português
* Русский
* کوردی
* Српски / srpski
* ไทย
* Українська
* Tiếng Việt
* 粵語
* 中文

Edit links

* Article
* Talk

English

* Read
* Edit
* View history

Tools

Tools

move to sidebar
hide

Actions

* Read
* Edit
* View history

General

* What links here
* Related changes
* Upload file
* Permanent link
* Page information
* Cite this page
* Get shortened URL
* Download QR code

Print/export

* Download as PDF
* Printable version

In other projects

* Wikidata item

Appearance

move to sidebar
hide

From Wikipedia, the free encyclopedia

Method in natural language processing

|  |
| --- |
| Part of a series on |
| Machine learning and data mining |
| Paradigms  * Supervised learning * Unsupervised learning * Semi-supervised learning * Self-supervised learning * Reinforcement learning * Meta-learning * Online learning * Batch learning * Curriculum learning * Rule-based learning * Neuro-symbolic AI * Neuromorphic engineering * Quantum machine learning |
| Problems  * Classification * Generative modeling * Regression * Clustering * Dimensionality reduction * Density estimation * Anomaly detection * Data cleaning * AutoML * Association rules * Semantic analysis * Structured prediction * Feature engineering * Feature learning * Learning to rank * Grammar induction * Ontology learning * Multimodal learning |
| Supervised learning (**classification** • **regression**)  * Apprenticeship learning * Decision trees * Ensembles   + Bagging   + Boosting   + Random forest * *k*-NN * Linear regression * Naive Bayes * Artificial neural networks * Logistic regression * Perceptron * Relevance vector machine (RVM) * Support vector machine (SVM) |
| Clustering  * BIRCH * CURE * Hierarchical * *k*-means * Fuzzy * Expectation–maximization (EM) * DBSCAN * OPTICS * Mean shift |
| Dimensionality reduction  * Factor analysis * CCA * ICA * LDA * NMF * PCA * PGD * t-SNE * SDL |
| Structured prediction  * Graphical models   + Bayes net   + Conditional random field   + Hidden Markov |
| Anomaly detection  * RANSAC * *k*-NN * Local outlier factor * Isolation forest |
| Artificial neural network  * Autoencoder * Deep learning * Feedforward neural network * Recurrent neural network   + LSTM   + GRU   + ESN   + reservoir computing * Boltzmann machine   + Restricted * GAN * Diffusion model * SOM * Convolutional neural network   + U-Net   + LeNet   + AlexNet   + DeepDream * Neural radiance field * Transformer   + Vision * Mamba * Spiking neural network * Memtransistor * Electrochemical RAM (ECRAM) |
| Reinforcement learning  * Q-learning * SARSA * Temporal difference (TD) * Multi-agent   + Self-play |
| Learning with humans  * Active learning * Crowdsourcing * Human-in-the-loop * RLHF |
| Model diagnostics  * Coefficient of determination * Confusion matrix * Learning curve * ROC curve |
| Mathematical foundations  * Kernel machines * Bias–variance tradeoff * Computational learning theory * Empirical risk minimization * Occam learning * PAC learning * Statistical learning * VC theory * Topological deep learning |
| Journals and conferences  * ECML PKDD * NeurIPS * ICML * ICLR * IJCAI * ML * JMLR |
| Related articles  * Glossary of artificial intelligence * List of datasets for machine-learning research   + List of datasets in computer vision and image processing * Outline of machine learning |
| * v * t * e |

Illustration of word embedding. Each word is a point in some space. The word embedding enables to perform semantic operator like obtaining the capital of a given country.

In natural language processing, a **word embedding** is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning.[1] Word embeddings can be obtained using language modeling and feature learning techniques, where words or phrases from the vocabulary are mapped to vectors of real numbers.

Methods to generate this mapping include neural networks,[2] dimensionality reduction on the word co-occurrence matrix,[3][4][5] probabilistic models,[6] explainable knowledge base method,[7] and explicit representation in terms of the context in which words appear.[8]

Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing[9] and sentiment analysis.[10]

## Development and history of the approach

[edit]

In distributional semantics, a quantitative methodological approach for understanding meaning in observed language, word embeddings or semantic feature space models have been used as a knowledge representation for some time.[11] Such models aim to quantify and categorize semantic similarities between linguistic items based on their distributional properties in large samples of language data. The underlying idea that "a word is characterized by the company it keeps" was proposed in a 1957 article by John Rupert Firth,[12] but also has roots in the contemporaneous work on search systems[13] and in cognitive psychology.[14]

The notion of a semantic space with lexical items (words or multi-word terms) represented as vectors or embeddings is based on the computational challenges of capturing distributional characteristics and using them for practical application to measure similarity between words, phrases, or entire documents. The first generation of semantic space models is the vector space model for information retrieval.[15][16][17] Such vector space models for words and their distributional data implemented in their simplest form results in a very sparse vector space of high dimensionality (cf. curse of dimensionality). Reducing the number of dimensions using linear algebraic methods such as singular value decomposition then led to the introduction of latent semantic analysis in the late 1980s and the random indexing approach for collecting word co-occurrence contexts.[18][19][20][21] In 2000, Bengio et al. provided in a series of papers titled "Neural probabilistic language models" to reduce the high dimensionality of word representations in contexts by "learning a distributed representation for words".[22][23][24]

A study published in NeurIPS (NIPS) 2002 introduced the use of both word and document embeddings applying the method of kernel CCA to bilingual (and multi-lingual) corpora, also providing an early example of self-supervised learning of word embeddings.[25]

Word embeddings come in two different styles, one in which words are expressed as vectors of co-occurring words, and another in which words are expressed as vectors of linguistic contexts in which the words occur; these different styles are studied in Lavelli et al., 2004.[26] Roweis and Saul published in *Science* how to use "locally linear embedding" (LLE) to discover representations of high dimensional data structures.[27] Most new word embedding techniques after about 2005 rely on a neural network architecture instead of more probabilistic and algebraic models, after foundational work done by Yoshua Bengio[28][*circular reference*] and colleagues.[29][30]

The approach has been adopted by many research groups after theoretical advances in 2010 had been made on the quality of vectors and the training speed of the model, as well as after hardware advances allowed for a broader parameter space to be explored profitably. In 2013, a team at Google led by Tomas Mikolov created word2vec, a word embedding toolkit that can train vector space models faster than previous approaches. The word2vec approach has been widely used in experimentation and was instrumental in raising interest for word embeddings as a technology, moving the research strand out of specialised research into broader experimentation and eventually paving the way for practical application.[31]

## Polysemy and homonymy

[edit]

Historically, one of the main limitations of static word embeddings or word vector space models is that words with multiple meanings are conflated into a single representation (a single vector in the semantic space). In other words, polysemy and homonymy are not handled properly. For example, in the sentence "The club I tried yesterday was great!", it is not clear if the term *club* is related to the word sense of a *club sandwich*, *clubhouse*, *golf club*, or any other sense that *club* might have. The necessity to accommodate multiple meanings per word in different vectors (multi-sense embeddings) is the motivation for several contributions in NLP to split single-sense embeddings into multi-sense ones.[32][33]

Most approaches that produce multi-sense embeddings can be divided into two main categories for their word sense representation, i.e., unsupervised and knowledge-based.[34] Based on word2vec skip-gram, Multi-Sense Skip-Gram (MSSG)[35] performs word-sense discrimination and embedding simultaneously, improving its training time, while assuming a specific number of senses for each word. In the Non-Parametric Multi-Sense Skip-Gram (NP-MSSG) this number can vary depending on each word. Combining the prior knowledge of lexical databases (e.g., WordNet, ConceptNet, BabelNet), word embeddings and word sense disambiguation, Most Suitable Sense Annotation (MSSA)[36] labels word-senses through an unsupervised and knowledge-based approach, considering a word's context in a pre-defined sliding window. Once the words are disambiguated, they can be used in a standard word embeddings technique, so multi-sense embeddings are produced. MSSA architecture allows the disambiguation and annotation process to be performed recurrently in a self-improving manner.[37]

The use of multi-sense embeddings is known to improve performance in several NLP tasks, such as part-of-speech tagging, semantic relation identification, semantic relatedness, named entity recognition and sentiment analysis.[38][39]

As of the late 2010s, contextually-meaningful embeddings such as ELMo and BERT have been developed.[40] Unlike static word embeddings, these embeddings are at the token-level, in that each occurrence of a word has its own embedding. These embeddings better reflect the multi-sense nature of words, because occurrences of a word in similar contexts are situated in similar regions of BERT’s embedding space.[41][42]

## For biological sequences: BioVectors

[edit]

Word embeddings for *n-*grams in biological sequences (e.g. DNA, RNA, and Proteins) for bioinformatics applications have been proposed by Asgari and Mofrad.[43] Named bio-vectors (BioVec) to refer to biological sequences in general with protein-vectors (ProtVec) for proteins (amino-acid sequences) and gene-vectors (GeneVec) for gene sequences, this representation can be widely used in applications of deep learning in proteomics and genomics. The results presented by Asgari and Mofrad[43] suggest that BioVectors can characterize biological sequences in terms of biochemical and biophysical interpretations of the underlying patterns.

## Game design

[edit]

Word embeddings with applications in game design have been proposed by Rabii and Cook[44] as a way to discover emergent gameplay using logs of gameplay data. The process requires transcribing actions that occur during a game within a formal language and then using the resulting text to create word embeddings. The results presented by Rabii and Cook[44] suggest that the resulting vectors can capture expert knowledge about games like chess that are not explicitly stated in the game's rules.

## Sentence embeddings

[edit]

Main article: Sentence embedding

The idea has been extended to embeddings of entire sentences or even documents, e.g. in the form of the thought vectors concept. In 2015, some researchers suggested "skip-thought vectors" as a means to improve the quality of machine translation.[45] A more recent and popular approach for representing sentences is Sentence-BERT, or SentenceTransformers, which modifies pre-trained BERT with the use of siamese and triplet network structures.[46]

## Software

[edit]

Software for training and using word embeddings includes Tomáš Mikolov's Word2vec, Stanford University's GloVe,[47] GN-GloVe,[48] Flair embeddings,[38] AllenNLP's ELMo,[49] BERT,[50] fastText, Gensim,[51] Indra,[52] and Deeplearning4j. Principal Component Analysis (PCA) and T-Distributed Stochastic Neighbour Embedding (t-SNE) are both used to reduce the dimensionality of word vector spaces and visualize word embeddings and clusters.[53]

### Examples of application

[edit]

For instance, the fastText is also used to calculate word embeddings for text corpora in Sketch Engine that are available online.[54]

## Ethical implications

[edit]

Word embeddings may contain the biases and stereotypes contained in the trained dataset, as Bolukbasi et al. points out in the 2016 paper “Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings” that a publicly available (and popular) word2vec embedding trained on Google News texts (a commonly used data corpus), which consists of text written by professional journalists, still shows disproportionate word associations reflecting gender and racial biases when extracting word analogies.[55] For example, one of the analogies generated using the aforementioned word embedding is “man is to computer programmer as woman is to homemaker”.[56][57]

Research done by Jieyu Zhou et al. shows that the applications of these trained word embeddings without careful oversight likely perpetuates existing bias in society, which is introduced through unaltered training data. Furthermore, word embeddings can even amplify these biases .[58][59]

## See also

[edit]

* Embedding (machine learning)
* Brown clustering
* Distributional–relational database

## References

[edit]

1. **^** Jurafsky, Daniel; H. James, Martin (2000). *Speech and language processing : an introduction to natural language processing, computational linguistics, and speech recognition*. Upper Saddle River, N.J.: Prentice Hall. ISBN 978-0-13-095069-7.
2. **^** Mikolov, Tomas; Sutskever, Ilya; Chen, Kai; Corrado, Greg; Dean, Jeffrey (2013). "Distributed Representations of Words and Phrases and their Compositionality". arXiv:1310.4546 [cs.CL].
3. **^** Lebret, Rémi; Collobert, Ronan (2013). "Word Emdeddings through Hellinger PCA". *Conference of the European Chapter of the Association for Computational Linguistics (EACL)*. Vol. 2014. arXiv:1312.5542.
4. **^** Levy, Omer; Goldberg, Yoav (2014). *Neural Word Embedding as Implicit Matrix Factorization* (PDF). NIPS.
5. **^** Li, Yitan; Xu, Linli (2015). *Word Embedding Revisited: A New Representation Learning and Explicit Matrix Factorization Perspective* (PDF). Int'l J. Conf. on Artificial Intelligence (IJCAI).
6. **^** Globerson, Amir (2007). "Euclidean Embedding of Co-occurrence Data" (PDF). *Journal of Machine Learning Research*.
7. **^** Qureshi, M. Atif; Greene, Derek (2018-06-04). "EVE: explainable vector based embedding technique using Wikipedia". *Journal of Intelligent Information Systems*. **53**: 137–165. arXiv:1702.06891. doi:10.1007/s10844-018-0511-x. ISSN 0925-9902. S2CID 10656055.
8. **^** Levy, Omer; Goldberg, Yoav (2014). *Linguistic Regularities in Sparse and Explicit Word Representations* (PDF). CoNLL. pp. 171–180.
9. **^** Socher, Richard; Bauer, John; Manning, Christopher; Ng, Andrew (2013). *Parsing with compositional vector grammars* (PDF). Proc. ACL Conf. Archived from the original (PDF) on 2016-08-11. Retrieved 2014-08-14.
10. **^** Socher, Richard; Perelygin, Alex; Wu, Jean; Chuang, Jason; Manning, Chris; Ng, Andrew; Potts, Chris (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank* (PDF). EMNLP.
11. **^** Sahlgren, Magnus. "A brief history of word embeddings".
12. **^** Firth, J.R. (1957). "A synopsis of linguistic theory 1930–1955". *Studies in Linguistic Analysis*: 1–32. Reprinted in F.R. Palmer, ed. (1968). *Selected Papers of J.R. Firth 1952–1959*. London: Longman.`{{cite book}}`: CS1 maint: publisher location (link)
13. **^** Luhn, H.P. (1953). "A New Method of Recording and Searching Information". *American Documentation*. **4**: 14–16. doi:10.1002/asi.5090040104.
14. **^** Osgood, C.E.; Suci, G.J.; Tannenbaum, P.H. (1957). *The Measurement of Meaning*. University of Illinois Press.
15. **^** Salton, Gerard (1962). "Some experiments in the generation of word and document associations". *Proceedings of the December 4-6, 1962, fall joint computer conference on - AFIPS '62 (Fall)*. pp. 234–250. doi:10.1145/1461518.1461544. ISBN 9781450378796. S2CID 9937095. `{{cite book}}`: ISBN / Date incompatibility (help)
16. **^** Salton, Gerard; Wong, A; Yang, C S (1975). "A Vector Space Model for Automatic Indexing". *Communications of the ACM*. **18** (11): 613–620. doi:10.1145/361219.361220. hdl:1813/6057. S2CID 6473756.
17. **^** Dubin, David (2004). "The most influential paper Gerard Salton never wrote". Archived from the original on 18 October 2020. Retrieved 18 October 2020.
18. **^** Kanerva, Pentti, Kristoferson, Jan and Holst, Anders (2000): Random Indexing of Text Samples for Latent Semantic Analysis, Proceedings of the 22nd Annual Conference of the Cognitive Science Society, p. 1036. Mahwah, New Jersey: Erlbaum, 2000.
19. **^** Karlgren, Jussi; Sahlgren, Magnus (2001). Uesaka, Yoshinori; Kanerva, Pentti; Asoh, Hideki (eds.). "From words to understanding". *Foundations of Real-World Intelligence*. CSLI Publications: 294–308.
20. **^** Sahlgren, Magnus (2005) An Introduction to Random Indexing, Proceedings of the Methods and Applications of Semantic Indexing Workshop at the 7th International Conference on Terminology and Knowledge Engineering, TKE 2005, August 16, Copenhagen, Denmark
21. **^** Sahlgren, Magnus, Holst, Anders and Pentti Kanerva (2008) Permutations as a Means to Encode Order in Word Space, In Proceedings of the 30th Annual Conference of the Cognitive Science Society: 1300–1305.
22. **^** Bengio, Yoshua; Réjean, Ducharme; Pascal, Vincent (2000). "A Neural Probabilistic Language Model" (PDF). *NeurIPS*.
23. **^** Bengio, Yoshua; Ducharme, Réjean; Vincent, Pascal; Jauvin, Christian (2003). "A Neural Probabilistic Language Model" (PDF). *Journal of Machine Learning Research*. **3**: 1137–1155.
24. **^** Bengio, Yoshua; Schwenk, Holger; Senécal, Jean-Sébastien; Morin, Fréderic; Gauvain, Jean-Luc (2006). "A Neural Probabilistic Language Model". *Studies in Fuzziness and Soft Computing*. Vol. 194. Springer. pp. 137–186. doi:10.1007/3-540-33486-6\_6. ISBN 978-3-540-30609-2.
25. **^** Vinkourov, Alexei; Cristianini, Nello; Shawe-Taylor, John (2002). *Inferring a semantic representation of text via cross-language correlation analysis* (PDF). Advances in Neural Information Processing Systems. Vol. 15.
26. **^** Lavelli, Alberto; Sebastiani, Fabrizio; Zanoli, Roberto (2004). *Distributional term representations: an experimental comparison*. 13th ACM International Conference on Information and Knowledge Management. pp. 615–624. doi:10.1145/1031171.1031284.
27. **^** Roweis, Sam T.; Saul, Lawrence K. (2000). "Nonlinear Dimensionality Reduction by Locally Linear Embedding". *Science*. **290** (5500): 2323–6. Bibcode:2000Sci...290.2323R. CiteSeerX 10.1.1.111.3313. doi:10.1126/science.290.5500.2323. PMID 11125150. S2CID 5987139.
28. **^** he:יהושע בנג'יו
29. **^** Morin, Fredric; Bengio, Yoshua (2005). "Hierarchical probabilistic neural network language model" (PDF). In Cowell, Robert G.; Ghahramani, Zoubin (eds.). *Proceedings of the Tenth International Workshop on Artificial Intelligence and Statistics*. Proceedings of Machine Learning Research. Vol. R5. pp. 246–252.
30. **^** Mnih, Andriy; Hinton, Geoffrey (2009). "A Scalable Hierarchical Distributed Language Model". *Advances in Neural Information Processing Systems*. 21 (NIPS 2008). Curran Associates, Inc.: 1081–1088.
31. **^** "word2vec". *Google Code Archive*. Retrieved 23 July 2021.
32. **^** Reisinger, Joseph; Mooney, Raymond J. (2010). *Multi-Prototype Vector-Space Models of Word Meaning*. Vol. Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics. Los Angeles, California: Association for Computational Linguistics. pp. 109–117. ISBN 978-1-932432-65-7. Retrieved October 25, 2019.
33. **^** Huang, Eric. (2012). *Improving word representations via global context and multiple word prototypes*. OCLC 857900050.
34. **^** Camacho-Collados, Jose; Pilehvar, Mohammad Taher (2018). "From Word to Sense Embeddings: A Survey on Vector Representations of Meaning". arXiv:1805.04032 [cs.CL].
35. **^** Neelakantan, Arvind; Shankar, Jeevan; Passos, Alexandre; McCallum, Andrew (2014). "Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space". *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Stroudsburg, PA, USA: Association for Computational Linguistics. pp. 1059–1069. arXiv:1504.06654. doi:10.3115/v1/d14-1113. S2CID 15251438.
36. **^** Ruas, Terry; Grosky, William; Aizawa, Akiko (2019-12-01). "Multi-sense embeddings through a word sense disambiguation process". *Expert Systems with Applications*. **136**: 288–303. arXiv:2101.08700. doi:10.1016/j.eswa.2019.06.026. hdl:2027.42/145475. ISSN 0957-4174. S2CID 52225306.
37. **^** Agre, Gennady; Petrov, Daniel; Keskinova, Simona (2019-03-01). "Word Sense Disambiguation Studio: A Flexible System for WSD Feature Extraction". *Information*. **10** (3): 97. doi:10.3390/info10030097. ISSN 2078-2489.
38. ^ ***a*** ***b*** Akbik, Alan; Blythe, Duncan; Vollgraf, Roland (2018). "Contextual String Embeddings for Sequence Labeling". *Proceedings of the 27th International Conference on Computational Linguistics*. Santa Fe, New Mexico, USA: Association for Computational Linguistics: 1638–1649.
39. **^** Li, Jiwei; Jurafsky, Dan (2015). "Do Multi-Sense Embeddings Improve Natural Language Understanding?". *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*. Stroudsburg, PA, USA: Association for Computational Linguistics. pp. 1722–1732. arXiv:1506.01070. doi:10.18653/v1/d15-1200. S2CID 6222768.
40. **^** Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina (June 2019). "Proceedings of the 2019 Conference of the North". *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Association for Computational Linguistics: 4171–4186. doi:10.18653/v1/N19-1423. S2CID 52967399.
41. **^** Lucy, Li, and David Bamman. "Characterizing English variation across social media communities with BERT." Transactions of the Association for Computational Linguistics 9 (2021): 538-556.
42. **^** Reif, Emily, Ann Yuan, Martin Wattenberg, Fernanda B. Viegas, Andy Coenen, Adam Pearce, and Been Kim. "Visualizing and measuring the geometry of BERT." Advances in Neural Information Processing Systems 32 (2019).
43. ^ ***a*** ***b*** Asgari, Ehsaneddin; Mofrad, Mohammad R.K. (2015). "Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics". *PLOS ONE*. **10** (11): e0141287. arXiv:1503.05140. Bibcode:2015PLoSO..1041287A. doi:10.1371/journal.pone.0141287. PMC 4640716. PMID 26555596.
44. ^ ***a*** ***b*** Rabii, Younès; Cook, Michael (2021-10-04). "Revealing Game Dynamics via Word Embeddings of Gameplay Data". *Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment*. **17** (1): 187–194. doi:10.1609/aiide.v17i1.18907. ISSN 2334-0924. S2CID 248175634.
45. **^** Kiros, Ryan; Zhu, Yukun; Salakhutdinov, Ruslan; Zemel, Richard S.; Torralba, Antonio; Urtasun, Raquel; Fidler, Sanja (2015). "skip-thought vectors". arXiv:1506.06726 [cs.CL].
46. **^** Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982-3992. 2019.
47. **^** "GloVe".
48. **^** Zhao, Jieyu; et al. (2018) (2018). "Learning Gender-Neutral Word Embeddings". arXiv:1809.01496 [cs.CL].
49. **^** "Elmo". 16 October 2024.
50. **^** Pires, Telmo; Schlinger, Eva; Garrette, Dan (2019-06-04). "How multilingual is Multilingual BERT?". arXiv:1906.01502 [cs.CL].
51. **^** "Gensim".
52. **^** "Indra". *GitHub*. 2018-10-25.
53. **^** Ghassemi, Mohammad; Mark, Roger; Nemati, Shamim (2015). "A visualization of evolving clinical sentiment using vector representations of clinical notes" (PDF). *2015 Computing in Cardiology Conference (CinC)*. Vol. 2015. pp. 629–632. doi:10.1109/CIC.2015.7410989. ISBN 978-1-5090-0685-4. PMC 5070922. PMID 27774487.
54. **^** "Embedding Viewer". *Embedding Viewer*. Lexical Computing. Archived from the original on 8 February 2018. Retrieved 7 Feb 2018.
55. **^** Bolukbasi, Tolga; Chang, Kai-Wei; Zou, James; Saligrama, Venkatesh; Kalai, Adam (2016). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings". arXiv:1607.06520 [cs.CL].
56. **^** Bolukbasi, Tolga; Chang, Kai-Wei; Zou, James; Saligrama, Venkatesh; Kalai, Adam (2016-07-21). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings". arXiv:1607.06520 [cs.CL].
57. **^** Dieng, Adji B.; Ruiz, Francisco J. R.; Blei, David M. (2020). "Topic Modeling in Embedding Spaces". *Transactions of the Association for Computational Linguistics*. **8**: 439–453. arXiv:1907.04907. doi:10.1162/tacl\_a\_00325.
58. **^** Zhao, Jieyu; Wang, Tianlu; Yatskar, Mark; Ordonez, Vicente; Chang, Kai-Wei (2017). "Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints". *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. pp. 2979–2989. doi:10.18653/v1/D17-1323.
59. **^** Petreski, Davor; Hashim, Ibrahim C. (2022-05-26). "Word embeddings are biased. But whose bias are they reflecting?". *AI & Society*. **38** (2): 975–982. doi:10.1007/s00146-022-01443-w. ISSN 1435-5655. S2CID 249112516.

| * v * t * e  Natural language processing | |
| --- | --- |
| General terms | * AI-complete * Bag-of-words * n-gram   + Bigram   + Trigram * Computational linguistics * Natural language understanding * Stop words * Text processing |
| Text analysis | * Argument mining * Collocation extraction * Concept mining * Coreference resolution * Deep linguistic processing * Distant reading * Information extraction * Named-entity recognition * Ontology learning * Parsing   + Semantic parsing   + Syntactic parsing * Part-of-speech tagging * Semantic analysis * Semantic role labeling * Semantic decomposition * Semantic similarity * Sentiment analysis  * Terminology extraction * Text mining * Textual entailment * Truecasing * Word-sense disambiguation * Word-sense induction   |  |  | | --- | --- | | Text segmentation | * Compound-term processing * Lemmatisation * Lexical analysis * Text chunking * Stemming * Sentence segmentation * Word segmentation | |
| Automatic summarization | * Multi-document summarization * Sentence extraction * Text simplification |
| Machine translation | * Computer-assisted * Example-based * Rule-based * Statistical * Transfer-based * Neural |
| Distributional semantics models | * BERT * Document-term matrix * Explicit semantic analysis * fastText * GloVe * Language model (large) * Latent semantic analysis * Seq2seq * Word embedding * Word2vec |
| Language resources, datasets and corpora | |  |  | | --- | --- | | Types and standards | * Corpus linguistics * Lexical resource * Linguistic Linked Open Data * Machine-readable dictionary * Parallel text * PropBank * Semantic network * Simple Knowledge Organization System * Speech corpus * Text corpus * Thesaurus (information retrieval) * Treebank * Universal Dependencies | | Data | * BabelNet * Bank of English * DBpedia * FrameNet * Google Ngram Viewer * UBY * WordNet * Wikidata | |
| Automatic identification and data capture | * Speech recognition * Speech segmentation * Speech synthesis * Natural language generation * Optical character recognition |
| Topic model | * Document classification * Latent Dirichlet allocation * Pachinko allocation |
| Computer-assisted reviewing | * Automated essay scoring * Concordancer * Grammar checker * Predictive text * Pronunciation assessment * Spell checker |
| Natural language user interface | * Chatbot * Interactive fiction (c.f. Syntax guessing) * Question answering * Virtual assistant * Voice user interface |
| Related | * Formal semantics * Hallucination * Natural Language Toolkit * spaCy |

| * v * t * e  Artificial intelligence (AI) | |
| --- | --- |
| History (timeline) | |
| Concepts | * Parameter   + Hyperparameter * Loss functions * Regression   + Bias–variance tradeoff   + Double descent   + Overfitting * Clustering * Gradient descent   + SGD   + Quasi-Newton method   + Conjugate gradient method * Backpropagation * Attention * Convolution * Normalization   + Batchnorm * Activation   + Softmax   + Sigmoid   + Rectifier * Gating * Weight initialization * Regularization * Datasets   + Augmentation * Prompt engineering * Reinforcement learning   + Q-learning   + SARSA   + Imitation   + Policy gradient * Diffusion * Latent diffusion model * Autoregression * Adversary * RAG * Uncanny valley * RLHF * Self-supervised learning * Recursive self-improvement * Word embedding * Hallucination |
| Applications | * Machine learning   + In-context learning * Artificial neural network   + Deep learning * Language model   + Large language model   + NMT * Artificial general intelligence (AGI) |
| Implementations | |  |  | | --- | --- | | Audio–visual | * AlexNet * WaveNet * Human image synthesis * HWR * OCR * Speech synthesis   + 15.ai   + ElevenLabs * Speech recognition   + Whisper * Facial recognition * AlphaFold * Text-to-image models   + Aurora   + DALL-E   + Firefly   + Flux   + Ideogram   + Imagen   + Midjourney   + Stable Diffusion * Text-to-video models   + Dream Machine   + Runway Gen   + Hailuo AI   + Kling   + Sora   + Veo * Music generation   + Suno AI   + Udio | | Text | * Word2vec * Seq2seq * GloVe * BERT * T5 * Llama * Chinchilla AI * PaLM * GPT   + 1   + 2   + 3   + J   + ChatGPT   + 4   + 4o   + o1   + o3   + 4.5   + 4.1   + o4 * Claude * Gemini   + chatbot * Grok * LaMDA * BLOOM * Project Debater * IBM Watson * IBM Watsonx * Granite * PanGu-Σ * DeepSeek * Qwen | | Decisional | * AlphaGo * AlphaZero * OpenAI Five * Self-driving car * MuZero * Action selection   + AutoGPT * Robot control | |
| People | * Alan Turing * Warren Sturgis McCulloch * Walter Pitts * John von Neumann * Claude Shannon * Marvin Minsky * John McCarthy * Nathaniel Rochester * Allen Newell * Cliff Shaw * Herbert A. Simon * Oliver Selfridge * Frank Rosenblatt * Bernard Widrow * Joseph Weizenbaum * Seymour Papert * Seppo Linnainmaa * Paul Werbos * Jürgen Schmidhuber * Yann LeCun * Geoffrey Hinton * John Hopfield * Yoshua Bengio * Lotfi A. Zadeh * Stephen Grossberg * Alex Graves * Andrew Ng * Fei-Fei Li * Alex Krizhevsky * Ilya Sutskever * Demis Hassabis * David Silver * Ian Goodfellow * Andrej Karpathy * James Goodnight |
| Architectures | * Neural Turing machine * Differentiable neural computer * Transformer   + Vision transformer (ViT) * Recurrent neural network (RNN) * Long short-term memory (LSTM) * Gated recurrent unit (GRU) * Echo state network * Multilayer perceptron (MLP) * Convolutional neural network (CNN) * Residual neural network (RNN) * Highway network * Mamba * Autoencoder * Variational autoencoder (VAE) * Generative adversarial network (GAN) * Graph neural network (GNN) |
| * Portals   + Technology * Category   + Artificial neural networks   + Machine learning * List   + Companies   + Projects | |

Retrieved from "https://en.wikipedia.org/w/index.php?title=Word\_embedding&oldid=1283071646"

Categories:

* Language modeling
* Artificial neural networks
* Natural language processing
* Computational linguistics
* Semantic relations

Hidden categories:

* CS1 maint: publisher location
* CS1 errors: ISBN date
* CS1: long volume value
* Articles with short description
* Short description matches Wikidata
* All articles lacking reliable references
* Articles lacking reliable references from May 2024