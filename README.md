# NLP-CLUSTERING

# NLP – Final Project: Analysis of Unrecognized User Requests in Goal-Oriented Dialog Systems


## Introduction

In this document, we explore a text clustering and title generation approach specifically designed for handling unrecognized requests. Unrecognized requests often contain unstructured text data, which poses challenges for efficient categorization and analysis. Our goal is to organize these requests by clustering similar ones together and generating descriptive titles for each cluster to provide insights into the unrecognized data.

## Understanding the Code

The provided code comprises several key components:

- **Library Import:** Essential libraries such as `json`, `numpy`, `pandas`, and modules from `sentence_transformers` and `nltk` are imported.
- **Constants and Parameters:** Constants like `MODEL_NAME` and clustering parameters are defined.
- **Sentence Transformer Model:** A Sentence Transformer model is initialized using the specified `MODEL_NAME`.

## Section 1: Clustering of Requests

### Approach

The clustering process groups similar requests based on their semantic similarity using dynamic means clustering. This technique iteratively updates cluster centroids and assigns data points to clusters until convergence is reached.

### Implementation Details

**Data Preprocessing:**
- The script loads textual data from a CSV file using the Pandas library.
- Text preprocessing includes removing special characters, converting text to lowercase, and tokenizing sentences.

**Embedding Generation:**
- Textual data is converted into numerical vectors using a pre-trained Sentence Transformer model.
- These embeddings capture the semantic meaning of sentences in a dense vector space.

**Dynamic Means Clustering:**
- Centroids are initialized as empty lists, and cluster assignments are managed using NumPy arrays.
- The algorithm iterates over data points, updating cluster centroids and assignments based on the Euclidean distance between data points and centroids.
- Convergence is achieved when cluster assignments remain unchanged between iterations or the maximum number of iterations is reached.

**Cluster Representation:**
- Clusters are represented as lists of sentences grouped by their assigned centroids.
- Sentences that do not meet the minimum cluster size requirement are stored separately as unclustered.

## Section 2: Generating Titles

### Approach

Title generation involves creating descriptive titles for each cluster to summarize the common theme or topic represented by the clustered requests. Titles are generated based on the most frequent 3-grams (sequences of three consecutive words) within each cluster.

### Implementation Details

**Text Processing:**
- Stop words and punctuation characters are removed to focus on meaningful content.
- Tokenization and generation of 3-grams are performed using the Natural Language Toolkit (NLTK) library.

**Title Extraction:**
- The most frequent 3-gram within each cluster is identified using the `Counter` class from the `collections` module.
- If no significant 3-gram is found, a default title is assigned.

**Output Formatting:**
- Titles and clustered requests are structured into a JSON format for easy readability and further analysis.

---

For any questions or further information, please open an issue in this repository.
