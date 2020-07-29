from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
import os
import re
import argparse
from os.path import join
import json
from tqdm import tqdm


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


def main(args):
    data_dir = args.data_dir
    split = args.split
    summary_size = args.summary_size
    omega = args.omega

    # make output dir
    os.makedirs(args.pred_path)
    os.makedirs(join(args.pred_path, 'output'))
    json.dump(vars(args), open(join(args.pred_path, 'log.json'), 'w'))

    split_dir = join(data_dir, split)
    n_data = _count_data(split_dir)
    documents = []  # a list of list of str
    queries = []  # a list of str
    for i in range(n_data):
    #for i in range(10):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        doc_sent_list = js['article']
        reference_entity_list = js["reference_entity_list_non_numerical"]
        if reference_entity_list:
            reference_entity_str = " ; ".join(reference_entity_list)
        else:
            reference_entity_str = ""
        documents.append(doc_sent_list)
        queries.append(reference_entity_str)

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    num_processed_doc = 0
    for doc_sent_list, query_str in tqdm(zip(documents, queries), total=len(documents)):
        #print("document:")
        #print(doc_sent_list)
        #print("query_str:")
        #print(query_str)
        query_focused_summary = lxr.get_query_focused_summary(doc_sent_list, query_str, summary_size, omega)
        #print("summary:")
        #print(query_focused_summary)
        #print()
        with open(join(args.pred_path, 'output/{}.dec'.format(num_processed_doc)), 'w') as f:
            f.write(make_html_safe('\n'.join(query_focused_summary)))
        num_processed_doc += 1

    # dump to json files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Query-focused summarization')
    )
    parser.add_argument('-data_dir', type=str, action='store', required=True,
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store', default="test",
                        help='train or val or test.')
    parser.add_argument('-pred_path', type=str, action='store', default="test",
                        help='path of output.')
    parser.add_argument('-summary_size', type=int, action='store', default=2,
                        help='number of sentences in output summary. ')
    parser.add_argument('-omega', type=int, action='store', default=6,
                        help='The diversity penalty parameter.')
    args = parser.parse_args()
    main(args)
