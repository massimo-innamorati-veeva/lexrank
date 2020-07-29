from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
import os
import re
import argparse
from os.path import join
import json


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def debug():
    omega = 0
    summary_size = 2

    documents = []
    documents_dir = Path('data/bbc/politics')
    for file_path in documents_dir.files('*.txt'):
        with file_path.open(mode='rt', encoding='utf-8') as fp:
            documents.append(fp.readlines())

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])
    #for document in documents:
    #    query_focused_summary = lxr.get_query_focused_summary(document, query, summary_size, omega)
    query = "George Osborne"
    #query = ""
    document = [
    'One of David Cameron\'s closest friends and Conservative allies, '
    'George Osborne rose rapidly after becoming MP for Tatton in 2001.',

    'Michael Howard promoted him from shadow chief secretary to the '
    'Treasury to shadow chancellor in May 2005, at the age of 34.',

    'Mr Osborne took a key role in the election campaign and has been at '
    'the forefront of the debate on how to deal with the recession and '
    'the UK\'s spending deficit.',

    'Even before Mr Cameron became leader the two were being likened to '
    'Labour\'s Blair/Brown duo. The two have emulated them by becoming '
    'prime minister and chancellor, but will want to avoid the spats.',

    'Before entering Parliament, he was a special adviser in the '
    'agriculture department when the Tories were in government and later '
    'served as political secretary to William Hague.',

    'The BBC understands that as chancellor, Mr Osborne, along with the '
    'Treasury will retain responsibility for overseeing banks and '
    'financial regulation.',

    'Mr Osborne said the coalition government was planning to change the '
    'tax system \"to make it fairer for people on low and middle '
    'incomes\", and undertake \"long-term structural reform\" of the '
    'banking sector, education and the welfare state.',
    ]
    query_focused_summary = lxr.get_query_focused_summary(document, query, summary_size, omega)
    print(query_focused_summary)


def main(data_dir, split, summary_size, omega):
    split_dir = join(data_dir, split)
    n_data = _count_data(split_dir)
    documents = []  # a list of list of str
    queries = []  # a list of str
    #for i in range(n_data):
    for i in range(10):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        doc_sent_list = js['article']
        reference_entity_list = js["reference_entity_list_non_numerical"]
        reference_entity_str = " ; ".join(reference_entity_list)
        documents.append(doc_sent_list)
        queries.append(reference_entity_str)
        print("document")
        print(doc_sent_list)
        print("query_str")
        print(reference_entity_str)
        exit()

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    for doc_sent_list, query_str in zip(documents, queries):
        print("document:")
        print(doc_sent_list)
        print("query_str:")
        print(query_str)
        query_focused_summary = lxr.get_query_focused_summary(doc_sent_list, query_str, summary_size, omega)
        print("summary:")
        print(query_focused_summary)
        print()

    # dump to json files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Query-focused summarization')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store',
                        help='train or val or test.')
    parser.add_argument('-summary_size', type=int, action='store',
                        help='number of sentences in output summary. ')
    parser.add_argument('-omega', type=str, action='store',
                        help='The diversity penalty parameter.')
    args = parser.parse_args()
    main(args.data_dir, args.split, summary_size=args.summary_size, omega=args.omega)
