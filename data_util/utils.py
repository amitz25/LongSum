#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
from sumeval.metrics.rouge import RougeCalculator
import logging
import tensorflow as tf

def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  rouge = RougeCalculator(stopwords=True, lang="en")
  rouge_1 = 0; rouge_2 = 0; rouge_3 = 0; rouge_l = 0
  num_files = len(os.listdir(ref_dir))
  for filename in os.listdir(ref_dir):
    file_id = filename[:6]
    ref_file = os.path.join(ref_dir, filename)
    dec_file = os.path.join(dec_dir, '%s_decoded.txt' % file_id)
    ref_sum = ' '.join([line for line in open(ref_file, 'r').readlines()])
    dec_sum = ' '.join([line for line in open(dec_file, 'r').readlines()])
    rouge_1 += rouge.rouge_n(dec_sum, ref_sum, n=1)
    rouge_2 += rouge.rouge_n(dec_sum, ref_sum, n=2)
    rouge_3 += rouge.rouge_n(dec_sum, ref_sum, n=3)
    rouge_l += rouge.rouge_l(dec_sum, ref_sum)
  return {'1': 100 * rouge_1 / num_files,
          '2': 100 * rouge_2 / num_files,
          '3': 100 * rouge_3 / num_files,
          'l': 100 * rouge_l / num_files}


def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1", "2", "3", "l"]:
    log_str += "\nROUGE-%s: %.4f\n" % (x, results_dict[x])
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  decoded_sents = [make_html_safe(w) for w in decoded_sents]
  reference_sents = [make_html_safe(w) for w in reference_sents]

  ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

  with open(ref_file, "w") as f:
    for idx, sent in enumerate(reference_sents):
      f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

  #print("Wrote example %i to file" % ex_index)
