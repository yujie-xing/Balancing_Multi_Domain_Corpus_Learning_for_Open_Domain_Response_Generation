import re
import pickle
import argparse
from collections import Counter
from os import path

parser = argparse.ArgumentParser()

parser.add_argument('--alpha', action='store_true',help='Turn on to generate alpha DF.')
parser.add_argument('--alpha_value', type=int, default=100)
parser.add_argument('--LSTM', action='store_true', help='Generate DF for LSTM model.')
parser.add_argument('--GPT', action='store_true', help='Generate DF for GPT model.')
parser.add_argument('--write', action='store_true', help='Turn on to write a new DF file.')
parser.add_argument('--file_path', type=str, default='DF_new.pkl')
parser.add_argument('--data_folder', type=str, default='../data')
parser.add_argument('--data_files', type=list, default=['test_OSDB.txt','test_Twitter.txt','test_Ubuntu.txt','test_PersonaChat.txt'])
parser.add_argument('--lstm_voc', type=str, default='vocab')

from nltk.tokenize import RegexpTokenizer
tok_alpha = RegexpTokenizer(r'\w+')
from pytorch_pretrained_bert import BertTokenizer,GPT2Tokenizer
tok_lstm=BertTokenizer.from_pretrained('bert-large-uncased')
tok_GPT=GPT2Tokenizer.from_pretrained('gpt2')

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def clear(text):

    """Lower text and remove punctuation and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(text)))
    return text


def calculate_DF(word_count,alpha,alpha_value):
    df_dict = {corpus:dict() for corpus in word_count}
    for corpus_name in word_count:
        other_names = {corpus for corpus in word_count}-{corpus_name}
        others = [word_count[other_name] for other_name in other_names]
        for word,count in word_count[corpus_name].most_common():
            if count==0:
                df = 0
            else:
                other_counts = 0
                all_in = 1
                for other in others:
                    try:
                        other_counts += other[word]
                    except KeyError:
                        pass
                df = count/(count+other_counts)
            df_dict[corpus_name][word] = df
        min_value = min(df_dict[corpus_name].values())
        max_value = max(df_dict[corpus_name].values())
        for word in df_dict[corpus_name]:
            df = df_dict[corpus_name][word]
            if alpha:
                df_dict[corpus_name][word] = alpha_value**((df-min_value)/(max_value-min_value))
            else:
                df_dict[corpus_name][word] = (df-min_value)/(max_value-min_value)
    return df_dict

def frequency(corpus):
    counts = Counter(corpus)
    total = sum(counts.values())
    min_value = min(counts.values())
    for word in counts:
        counts[word] = ((counts[word]-min_value)*100000.0)/total
    return counts


def DF_gen(args):
    if args.LSTM:
        lstm_voc = dict()
        with open(path.join(args.data_folder,args.lstm_voc),'r') as v:
            for line in v:
                lstm_voc[line.strip()]=len(lstm_voc)

    osdb = []
    twitter = []
    ubuntu = []
    personachat = []
    for doc in args.data_files:
        with open(path.join(args.data_folder,doc),'r') as d:
            for line in d:
                line = line.strip().split('|')
                if args.alpha:
                    tokens = tok_alpha.tokenize(line[0].lower())+tok_alpha.tokenize(line[1].lower())
                elif args.LSTM:
                    tokens = tok_lstm.tokenize(clear(line[-1]))
                    ids = []
                    for token in tokens:
                        try:
                            ids.append(lstm_voc[token]+3)
                        except:
                            pass
                    tokens = ids
                elif args.GPT:
                    tokens = tok_GPT.encode(clear(line[-1]))
                else:
                    raise Exception('Error: Choose from alpha, LSTM or GPT.')
                if 'OSDB' in doc:
                    osdb.extend(tokens)
                elif 'Twitter' in doc:
                    twitter.extend(tokens)
                elif 'Ubuntu' in doc:
                    ubuntu.extend(tokens)
                elif 'PersonaChat' in doc:
                    personachat.extend(tokens)
                else:
                    raise Exception('Error: document name fault')
    word_count = {'osdb':frequency(osdb),'twitter':frequency(twitter),'ubuntu':frequency(ubuntu),'convai':frequency(personachat)}
    df_dict = calculate_DF(word_count,args.alpha,args.alpha_value)

    if args.write:
        with open(args.file_path,'wb') as file:
            pickle.dump(df_dict,file)




if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    DF_gen(args)
    with open('DF_new.pkl','rb') as pkl:
        df = pickle.load(pkl)

    print(len(df['osdb'].keys()))
    print(len(df['ubuntu'].keys()))

    for word in ['i','to','laptop','file','ubuntu','music','hobby','hiking']:
        print(word)
        try:
            print(df['ubuntu'][word])
        except:
            print('not appeared in ubuntu')
        try:
            print(df['convai'][word])
        except:
            print('not appeared in convai')
        print('=========')