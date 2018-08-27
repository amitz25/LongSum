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

NUM_SENTENCES = 10
input_path = os.path.join('.', 'data_util', 'arxiv-release')

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

def get_masks(abstract_sentences, article_sentences, abstract_lengths, article_lengths):    
    #mask = np.zeros(len(article_sentences))
    masks = []

    with tf.Session() as sess:
        print("BEFORE MODEL")
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        articles_abstracts_embed = sess.run(embed(article_sentences + abstract_sentences))
        print("AFTER MODEL")
        num_article_sents = sum(article_lengths)
        articles_embed = articles_abstracts_embed[:num_article_sents, :]
        abstract_embed = articles_abstracts_embed[num_article_sents:, :]
        article_sent_ind = 0
        abstract_sent_ind = 0
        print("BEFORE INNER")
        for i in range(len(article_lengths)): #number of articles
            cur_article = articles_embed[article_sent_ind : article_sent_ind + article_lengths[i], :]
            cur_abstract = abstract_embed[abstract_sent_ind : abstract_sent_ind + abstract_lengths[i], :]
            article_sent_ind += article_lengths[i]
            abstract_sent_ind += abstract_lengths[i]
            #import pdb; pdb.set_trace()
            cur_mask = [0 for i in range(article_lengths[i])]
            for sent in range(cur_abstract.shape[0]):
                similarities = np.inner(cur_abstract[sent, :], cur_article)
                top_k = np.argsort(similarities)[-NUM_SENTENCES:].astype(int)#.tolist()
                for index in top_k:
                    cur_mask[index] = 1
            masks.append(cur_mask)
        print("AFTER INNER")

    return masks


def start():
    #import pdb; pdb.set_trace()
    with_mask_path = os.path.join(input_path, 'with_mask')
    num_chunked_files = len([name for name in os.listdir(os.path.join(input_path, 'chunked')) if os.path.isfile(os.path.join(input_path, 'chunked', name))])
    if not os.path.exists(with_mask_path):
        os.mkdir(with_mask_path)

    for file in range(num_chunked_files):
        #if file % 100 == 0:
        if True:
            print("finished %s files out of %s" %(file ,num_chunked_files))
        f = open(os.path.join(input_path, 'chunked', 'train_%s.txt' % file), 'r')
        open(os.path.join(with_mask_path, 'train_%s.txt' % file), 'a')
        out_f = open(os.path.join(with_mask_path, 'train_%s.txt' % file), 'w')
        lines = []
        article_lengths = []
        abstract_lengths = []
        abstract_sentences = []
        article_sentences = []
        for line in f.readlines():
            parsed_line = json.loads(line)
            abstract_sentences += parsed_line['abstract_text']
            article_sentences += parsed_line['article_text']           
            article_lengths.append(len(parsed_line['article_text']))
            abstract_lengths.append(len(parsed_line['abstract_text']))            
            lines.append(parsed_line)
            #import pdb; pdb.set_trace()
            
        string_lines = []
        masks = get_masks(abstract_sentences, article_sentences, abstract_lengths, article_lengths)
        for i in range(len(lines)):
            line = lines[i]
            line['sentence_mask'] = masks[i]
            string_lines.append(json.dumps(line) + '\n')
        
        out_f.write(''.join(string_lines))

start()
