from pytorch_pretrained_bert import GPT2LMHeadModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default='weight')
parser.add_argument('--data_folder', type=str, default='mix',
					help='folder for training & validation & test sets')
parser.add_argument('--test_set', type=str, default='test_opensubtitles1.txt,test_twitter1.txt,test_ubuntu1.txt,test_convai.txt')

parser.add_argument('--model_folder', type=str, default='trained_model/convai_GPT')
parser.add_argument('--model_name', type=str, default='model')

parser.add_argument('--output_folder', type=str, default='output')

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--cls_train", action="store_true")


args = parser.parse_args()
print(args)

if args.method == "mtl" or "weight":
	from GPT1 import *
else:
	from GPT import *

model=GPTdecoder(args)
model.decode()
# model.test()
