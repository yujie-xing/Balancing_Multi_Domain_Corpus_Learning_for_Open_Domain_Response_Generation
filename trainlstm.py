import argparse
from os import path

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default='weight')
parser.add_argument('--data_folder', type=str, default='data',
					help='folder for training & validation sets')
parser.add_argument('--df_file', type=str, default='DF_lstm.pkl')
parser.add_argument('--train_set', type=str, default='4-corpora_interleaved.txt')
# parser.add_argument('--train_set', type=str, default='train_OSDB.txt,train_Twitter.txt,train_Ubuntu.txt,train_PersonaChat.txt')
parser.add_argument('--valid_set', type=str, default='test_OSDB.txt,test_Twitter.txt,test_Ubuntu.txt,test_PersonaChat.txt')
parser.add_argument('--vocab', type=str, default='vocab')
parser.add_argument('--save_folder', type=str, default='trained_model/lstm')
parser.add_argument("--cls_train", action='store_true')


parser.add_argument("--max_iter", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=50)

parser.add_argument("--lr", type=float, default=1)
parser.add_argument("--dimension", type=int, default=512)
parser.add_argument("--layer", type=int, default=4)
parser.add_argument("--init_weight", type=float, default=0.1)

parser.add_argument('--max_grad_norm', type=int, default=5)
parser.add_argument('--start_half', type=int, default=4)

parser.add_argument("--vocab_num", type=, default=50000)

parser.add_argument('--ordered', action='store_true')
parser.add_argument('--fine_tune_path', type=str, default='trained_model/lstm_even')
parser.add_argument('--fine_tune_model', type=str, default='lstmmodel')
parser.add_argument('--fine_tune', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--optim', type=str, default='sgd',
					help='optimizing type: origin/adam/sgd')
parser.add_argument('--embedding', type=str, default='',
					help='Embedding type: bert/gpt/gpt2/path-to-embedding')
parser.add_argument('--fine_tune_turn', type=int, default=0)

args = parser.parse_args()
args.fine_tune_path=path.join(args.fine_tune_path,args.fine_tune_model)
print(args)
print()


if args.method == "mtl" or "weight":
	from lstm1 import *
else:
	from lstm import *

model=seq2seq(args)
model.train()