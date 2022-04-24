import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default='gpt2')
parser.add_argument('--method', type=str, default='weight')
parser.add_argument('--data_folder', type=str, default='data',
					help='folder for training & validation sets')
parser.add_argument('--df_file', type=str, default='DF_GPT.pkl')
parser.add_argument('--train_set', type=str, default='4-corpora_interleaved.txt')
# parser.add_argument('--train_set', type=str, default='train_OSDB.txt,train_Twitter.txt,train_Ubuntu.txt,train_PersonaChat.txt')
parser.add_argument('--valid_set', type=str, default='test_OSDB.txt,test_Twitter.txt,test_Ubuntu.txt,test_PersonaChat.txt')
parser.add_argument('--save_folder', type=str, default='trained_model/GPT')
parser.add_argument('--cls_train', action='store_true')

parser.add_argument("--max_iter", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=50)

parser.add_argument("--lr", type=float, default=6e-5)
parser.add_argument("--optimizer", type=str, default="GPT")
parser.add_argument("--dimension", type=int, default=768)
parser.add_argument("--layer", type=int, default=12)
parser.add_argument("--head", type=int, default=12)
parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5)
parser.add_argument("--initial_std", type=float, default=0.02)

parser.add_argument('--warmup_proportion', type=float, default=0.002)
parser.add_argument('--max_grad_norm', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.01)

parser.add_argument("--end_id", type=int, default=50256)
parser.add_argument("--n_special", type=int, default=2)
parser.add_argument("--n_token", type=int, default=2)

parser.add_argument('--ordered', action='store_true')
parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
parser.add_argument('--fine_tune_folder', type=str, default='trained_model/GPT_even')
parser.add_argument('--fine_tune_model', type=str, default='model')
parser.add_argument('--fine_tune', action='store_true')
parser.add_argument('--fine_tune_turn', type=int, default=0)
parser.add_argument("--no_gpt", action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--cpu', action='store_true')

args = parser.parse_args()
print(args)
print()

if args.method == "mtl" or "weight":
	from GPT1 import *
else:
	from GPT import *

model=GPT(args)
model.train()