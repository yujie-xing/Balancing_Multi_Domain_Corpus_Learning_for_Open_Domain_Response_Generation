import argparse
from os import path

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default='weight')
parser.add_argument('--data_folder', type=str, default='data',
					help='folder for training & validation & test sets')
parser.add_argument('--test_set', type=str, default='test_OSDB.txt,test_Twitter.txt,test_Ubuntu.txt,test_PersonaChat.txt')

parser.add_argument('--model_folder', type=str, default='trained_model/lstm')
parser.add_argument('--model_name', type=str, default='lstmmodel')

parser.add_argument('--output_folder', type=str, default='output')

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--layer", type=int, default=4)
parser.add_argument("--dimension", type=int, default=512)
parser.add_argument("--init_weight", type=float, default=0.1)	
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--cls_train", action="store_true")



args = parser.parse_args()
args.model=path.join(args.model_folder,args.model_name)
print(args)

if args.method == "mtl" or "weight":
	from lstmdecode1 import *
else:
	from lstmdecode import *

model=lstmdecoder(args)
model.decode()
# model.test()