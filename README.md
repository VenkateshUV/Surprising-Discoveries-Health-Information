# Surprising-Discoveries-Health-Information
Surprising Discoveries for Online Health Information

Tools Used: Python, Jupyter Notebook, genism, tensorflow 

• A pre-defined word vector trained on medical documents was obtained and cosine similarity between the sentence vectors were calculated.

• An API based on distributional similarity, Latent Semantic Analysis and semantic relations extracted from wordnet was used to find the similarity between two sentences, 25% of maximum cosine similarity in a document was set as threshold and surprising scores were calculated.

• An LDA model was constructed top 20 topics were extracted; also Word2Vec training was done to come up with word embeddings.
Techniques Used: Exploratory Data Analysis, Cosine Similarity, Latent Dirichlet Allocation, Latent Semantic Analysis & Word Embedding.

The source code for this project can be found in the Src directory.

API request.py - File for making the web request to API service and get the similarity score of sentences.

Cosine similarity.py - File for getting the cosine similarity of two sentences using pretrained word vectors and write output into csv file.

Phrase.py - separate file for calculating cosine similarity Tsne.png - Word embeddings visualization.

wordtoVec.ipynb - The python notebook for generating wordtovec model.

Topicmodeling.py - Python file for doing topic modeling.
