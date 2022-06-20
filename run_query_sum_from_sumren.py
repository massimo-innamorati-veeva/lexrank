from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
import os
import re
import argparse
from os.path import join
import json
from tqdm import tqdm
import jsonlines


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(args):
    filename = args.filenmae
    summary_size = args.summary_size
    omega = args.omega
    # make output dir
    os.makedirs(args.pred_path)
    #os.makedirs(join(args.pred_path, 'output'))
    #json.dump(vars(args), open(join(args.pred_path, 'log.json'), 'w'))

    documents = []  # a list of list of str
    queries = []  # a list of str
    # out_sample = {"text": " ".join(text_list), "query": query, "target": reference_summary}
    with jsonlines.open(filename) as f:
        for line_i, line in enumerate(tqdm(f)):
            doc_sent_list = line["text_sent_list"]
            query = line["query"]
            documents.append(doc_sent_list)
            queries.append(query)

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    num_processed_doc = 0
    out_summaries = []
    for doc_sent_list, query_str in tqdm(zip(documents, queries), total=len(documents)):
        #print("document:")
        #print(doc_sent_list)
        #print("query_str:")
        #print(query_str)
        query_focused_summary = lxr.get_query_focused_summary(doc_sent_list, query_str, summary_size, omega)
        #print("summary:")
        #print(query_focused_summary)
        #print()
        out_summaries.append(query_focused_summary)
        num_processed_doc += 1

    with open(join(args.pred_path, "output.txt"), "w") as f_out:
        f_out.write('\n'.join(out_summaries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Query-focused summarization')
    )
    parser.add_argument('-filenmae', type=str, action='store', required=True,
                        help='The directory of the data.')
    parser.add_argument('-pred_path', type=str, action='store', default="test",
                        help='path of output.')
    parser.add_argument('-summary_size', type=int, action='store', default=2,
                        help='number of sentences in output summary. ')
    parser.add_argument('-omega', type=int, action='store', default=4,
                        help='The diversity penalty parameter.')
    args = parser.parse_args()
    main(args)
