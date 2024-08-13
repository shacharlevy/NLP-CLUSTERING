import json
import numpy as np
import pandas as pd
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter


MODEL_NAME = 'all-MiniLM-L6-v2'
parameters = {
    'K': 56,
    'max_iterations': 30,
    'distance_threshold': 0.88
}
model = SentenceTransformer(MODEL_NAME)


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    df = load_data(data_file)
    sentences = df['text'].tolist()
    # processing the sentences - remove some charchters and convert to lower case
    sentences = [sentence.strip('\r\n').lower() for sentence in sentences]
    # encoding from sentences to vectors
    embeddings = model.encode(sentences)

    # clusters, centroids = dynamic_means_clustering(sentences, embeddings, parameters['distance_threshold'],
    #                               parameters['max_iterations'])
    # # write to the file like the structures in the example
    clusters, centroids = k_means_clustering(sentences, embeddings, parameters['K'], parameters['distance_threshold'],
                                             parameters['max_iterations'])
    write_to_file(clusters, output_file, min_size)
    # titled_clusters = {}
    # for i in range(parameters['K']):
    #     if len(clusters[i]) >= int(min_size):
    #         topic_sentence = get_title(clusters[i])
    #         titled_clusters[topic_sentence] = clusters[i]
    #
    # with open(output_file, 'w') as f:
    #     json.dump(titled_clusters, f)


def k_means_clustering(sentences, embeddings, k, distance_threshold, max_iterations=100):
    centroids = embeddings[np.random.choice(range(len(sentences)), k, replace=False)]

    for _ in range(max_iterations):
        cluster_assignment = []
        for emb in embeddings:
            distances = np.linalg.norm(centroids - emb, axis=1)
            min_distance = np.min(distances)
            # unclustered point
            if min_distance > distance_threshold:
                cluster_assignment.append(None)
            else:
                # add the index of the cenroid with minimum distance
                cluster_assignment.append(np.argmin(distances))

        # update centroids
        new_centroids = np.zeros_like(centroids)
        clusters_sizes = np.zeros(k)
        for i, emb in enumerate(embeddings):
            cluster = cluster_assignment[i]
            if cluster is not None:
                new_centroids[cluster] += emb
                clusters_sizes[cluster] += 1
        for i in range(k):
            if clusters_sizes[i] != 0:
                new_centroids[i] /= clusters_sizes[i]

        # check for convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    # assign sentences to clusters
    clusters = [[] for _ in range(k)]

    for i, sentence in enumerate(sentences):
        cluster = cluster_assignment[i]
        if cluster is not None:
            clusters[cluster].append(sentence)
    return clusters, new_centroids


def write_to_file(clusters, output_file, min_size):
    titled_clusters = {"cluster_list": []}
    unclustered = []
    for i in range(len(clusters)):
        if len(clusters[i]) >= int(min_size):
            title = get_title(clusters[i])
            cluster_dict = {"cluster_name": title, "requests": clusters[i]}
            titled_clusters["cluster_list"].append(cluster_dict)
        else:
            unclustered.extend(clusters[i])
    print(len(titled_clusters["cluster_list"]))
    titled_clusters["unclustered"] = unclustered


    with open(output_file, 'w') as f:
        json.dump(titled_clusters, f, indent=4)


# this function generate the titles for each cluster by finding the most
# frequent 3grams in the sentences of the cluster
def get_title(sentences):
    words = nltk.word_tokenize(" ".join(sentences).lower())

    # remove stop words
    stop_words = set(stopwords.words('english'))
    stop_characters = stop_words.union({"?", ".", ",", "'", "n't"})
    filtered_words = [word for word in words if word not in stop_characters]

    # generating 3-grams
    ngrams = zip(*[filtered_words[i:] for i in range(3)])
    three_grams = [" ".join(tuple(dict.fromkeys(ngram))) for ngram in ngrams]

    # count the instanes of each 3-grams
    three_gram_counts = Counter(three_grams)

    # get the top 3-grams
    top_three_grams = three_gram_counts.most_common(1)
    # just check if there isn't 3-gram to found
    if len(top_three_grams) > 0:
        return top_three_grams[0][0]
    else:
        return "no title found"


# this function generates the clusters
def dynamic_means_clustering(sentences, embeddings, distance_threshold, max_iterations=100):
    centroids = []
    # the assignment of the cluster for each vector in the previous iteration
    cluster_assignment_prev = np.array([-1] * len(embeddings))
    for _ in range(max_iterations):
        # the assignment of the cluster for each vector in the current iteration
        cluster_assignment = np.array([-1] * len(embeddings))
        shuffled_indices = np.random.permutation(len(embeddings))
        for i in shuffled_indices:
            emb = embeddings[i]
            # if this is the first vector we want to cluster (there isn't centroids)
            if len(centroids) == 0:
                centroids.append(emb)
                cluster_assignment[i] = 0
            else:
                # calculate the distance as Euclidian distance and takes the minimus disance from
                # the centroids
                distances = np.linalg.norm(np.array(centroids) - emb, axis=1)
                min_dist = np.min(distances)
                # if the minimum distance is bigger than the threshold
                # we put the vector as a new centroid
                if min_dist > distance_threshold:
                    centroids.append(emb)
                    cluster_assignment[i] = len(centroids) - 1
                else:
                    # we put the vector to the most close centroid to him
                    min_ind = np.argmin(distances)
                    cluster_assignment[i] = min_ind
        # if there is no changes for the assignments to the clusters we stop the iteration
        if np.array_equal(cluster_assignment_prev, cluster_assignment):
            break

        cluster_assignment_prev = cluster_assignment
        centroids = update_centroids(centroids, cluster_assignment, embeddings)

    clusters = make_cluster_sentences(sentences, cluster_assignment, centroids)

    return clusters, centroids


# group the original sentences by clusters
def make_cluster_sentences(sentences, cluster_assignment, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for i, sentence in enumerate(sentences):
        cluster = cluster_assignment[i]
        if cluster != -1:
            clusters[cluster].append(sentence)
    return clusters


# this function update the centroids after the iteration by calculating the mean of vectors in each cluster
def update_centroids(centroids, cluster_assignment, embeddings):
    new_centroids = []
    for i in range(len(centroids)):
        if not len(embeddings[cluster_assignment == i]) == 0:
            new_centroids.append(np.mean(embeddings[cluster_assignment == i], axis=0))

    return new_centroids





# load the data from the csv file using pandas library
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df





if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    # evaluate_clustering(config['example_solution_file'], config['output_file'])
