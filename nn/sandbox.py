from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
#import nltk
#from nltk.corpus import stopwords

vocab = Vocab(config.vocab_path, config.vocab_size)
batcher = Batcher(config.train_data_path, vocab, mode='train', batch_size=config.batch_size, single_pass=False)
batches = 1

def google_encoder_metric(abstract_sents, article_sents):
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    rotation = 90

    flatten = lambda l: [item for article_sents in l for item in article_sents]
    article_sentences = flatten(article_sents)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        #article_sentences = article_sentences[:3]
        #abstract_sents = abstract_sents[:2]

        article_embed = sess.run(embed(article_sentences))
        abstract_embed = sess.run(embed(abstract_sents))

        corr = np.inner(abstract_embed, article_embed)
        max_ind = np.argmax(corr, axis=1)

        for i, ab_s in enumerate(abstract_sents):
            print('from abstract:')
            print(ab_s)
            print('from article:')
            print(article_sentences[max_ind[i]])

        sns.set(font_scale=1.2)

        f, ax = plt.subplots(figsize=(10, 10))

        g = sns.heatmap(
            corr,
            xticklabels=abstract_sents,
            yticklabels=article_sentences,
            vmin=0,
            vmax=1,
            cmap="YlOrRd",
            ax=ax)
        plt.show()
        #fig = g.get_figure()
        #fig.savefig('fig')
        #g.set_xticklabels(abstract_sents, rotation=rotation)
        #g.set_title("Semantic Textual Similarity")


def bigrams_metric(abstract_sents, article_sents):
    """
    # Set up a quick lookup table for common words like "the" and "an" so they can be excluded
stops = set(stopwords.words('english'))

# Filter out words that have punctuation and make everything lower-case
cleaned_words = [w.lower() for w in word_list if w.isalnum()]

# Ask NLTK to generate a list of bigrams for the word "sun", excluding
# those words which are too common to be interesing
sun_bigrams = [b for b in nltk.bigrams(cleaned_words) if (b[0] == 'sun' or b[1] == 'sun') \
  and b[0] not in stops and b[1] not in stops]
    """
    print('___________')
    print('bigrams_metric')
    #print('filtered {0} sents out of {1}'.format(len(removed_setns), len(article_sents)))
    #return filtered_article
    return None

def run_all_metrics(abstract_sents, article_sents):
    #bigram_filtered_article = bigrams_metric(abstract_sents, article_sents)
    google_encoder_metric(abstract_sents, article_sents)

def start():
    for i_batch in np.arange(batches):
        print('##############')
        print('==== batch {0} ==='.format(i_batch))
        batch = batcher.next_batch()

        for i_input in np.arange(config.batch_size):
            print('==== input {0} ==='.format(i_input))
            abstract_sentences = batch.original_abstracts_sents[i_input]
            # article = batch.original_articles[i_input]
            article_sentences = batch.original_article_sents[i_input]
            run_all_metrics(abstract_sentences, article_sentences)

start()

"""
file_name = '../data_util/arxiv-release/chunked/train_0.txt'
reader = open(file_name, 'r')
lines = reader.read().splitlines()
line = lines[0]
line = json.loads(line)
x = 1
"""