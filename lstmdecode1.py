from os import path
import string
import numpy as np
import pickle
import linecache
import math

import torch
from torch import tensor
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from pytorch_pretrained_bert import BertTokenizer

class encoder():
	def __init__(self,dictionary):
		self.dic=dict()
		with open(dictionary,'r') as doc:
			for line in doc:
				self.dic[line.strip()]=len(self.dic)
		self.num=len(self.dic)

	def encode(self,tokens):
		ids=[]
		for token in tokens:
			try:
				ids.append(self.dic[token]+3)
			except KeyError:
				ids.append(self.num+3)
		return ids

class decoder():
	def __init__(self,dictionary):
		self.dic=dict()
		with open(dictionary,'r') as doc:
			for line in doc:
				self.dic[len(self.dic)]=line.strip()
		self.dic[len(self.dic)]='[NOK]'

	def decode(self,ids):
		tokens=""
		for idn in ids:
			try:
				word = self.dic[idn-3]
				if '##' in word:
					tokens += word[2:]
				else:
					tokens += " "+word
			except KeyError:
				break
		return tokens[1:]

class data():

	def __init__(self, batch_size, encoder, max_length=50):
		self.batch_size=batch_size
		self.max_length=max_length
		self.tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')
		self.encoder=encoder

	def train_batch(self, file, num):
		origin = []
		sources=np.zeros((self.batch_size, self.max_length+2))
		targets=np.zeros((self.batch_size, self.max_length+2))
		cls_label=-np.ones(self.batch_size)
		l_s_set=set()
		l_t_set=set()
		END=0
		a=0
		for i in range(self.batch_size):
			line=linecache.getline(file,num*self.batch_size+i+1).strip().split("|")
			i-=a
			if line==[""]:
				END=1
				break
			s=self.tokenizer.tokenize(line[-2])[:self.max_length]
			t=self.tokenizer.tokenize(line[-1])[:self.max_length]
			source=[1]+self.encoder.encode(s)+[2]
			target=[1]+self.encoder.encode(t)+[2]
			l_s=len(source)
			l_t=len(target)
			if l_s<3 or l_t<3:
				a+=1
				continue
			l_s_set.add(l_s)
			l_t_set.add(l_t)
			origin.append(line[-2])
			sources[i, :l_s]=source
			targets[i, :l_t]=target
			try:
				cls_label[i]=int(line[0])
			except:
				pass
			i+=1
		try:
			max_l_s=max(l_s_set)
			max_l_t=max(l_t_set)
		except ValueError:
			return END,None,None,None,None,None,None,None,None
		sources=sources[:i, : max_l_s]
		targets=targets[:i, : max_l_t]
		cls_label=cls_label[:i]
		mask_s=np.ones(sources.shape)*(sources!=0)
		mask_t=np.ones(targets.shape)*(targets!=0)
		length_s=(sources!=0).sum(1)
		token_num=mask_t[:,1:].sum()

		return END,tensor(sources).long(),tensor(targets).long(),tensor(cls_label).long(),tensor(mask_s).long(),tensor(mask_t).long(),tensor(length_s).long(),token_num,origin
	
	def cls_batch(self, file, num):
		sources=np.zeros((self.batch_size, self.max_length+1))
		l_set=set()
		END=0
		a=0
		for i in range(self.batch_size):
			line=linecache.getline(file,num*self.batch_size+i+1).strip().split("|")
			if line==[""]:
				END=1
				i-=a
				break
			s=self.tokenizer.tokenize(line[0])[:self.max_length]
			source=self.tokenizer.convert_tokens_to_ids(s)
			l=len(source)
			if l<1:
				a+=1
				i-=a
				continue
			l_set.add(l)
			i-=a
			sources[i, :l]=source
			i+=1
		try:
			max_l=max(l_set)
		except ValueError:
			return END,None,None
		sources=sources[:i, :max_l]
		mask=np.ones(sources.shape)*(sources!=0)
		if END==1:
			return END,tensor(sources[:i]).long(),tensor(mask[:i]).long()
		return END,tensor(sources).long(),tensor(mask).long()




class attention_feed(nn.Module):
	
	def __init__(self,dim):
		super(attention_feed, self).__init__()
	
	def forward(self,target_t,context):
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		return context_combined

class softattention(nn.Module):
	
	def __init__(self,dim):
		super(softattention, self).__init__()
		self.attlinear=nn.Linear(dim*2,dim,False)
	
	def forward(self,target_t,context):
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		output=self.attlinear(torch.cat((context_combined,target_t),-1))
		output=nn.Tanh()(output)
		return output


class lstm_source(nn.Module):
	
	def __init__(self,layer,dim,dropout,n_cls):
		super(lstm_source, self).__init__()
		self.lstms=nn.LSTM(dim,dim,num_layers=layer,batch_first=True,bias=False,dropout=dropout)
		if n_cls:
			self.cls_softlinear=nn.Linear(dim,n_cls,False)
			
	def forward(self,embedding,length,cls_label):
		packed=pack_padded_sequence(embedding,length,batch_first=True,enforce_sorted=False)
		packed_output,(h,c)=self.lstms(packed)
		context,_= pad_packed_sequence(packed_output,batch_first=True)
		if cls_label is not None:
			cls_label_pred = self.cls_softlinear(context.sum(1))
		else:
			cls_label_pred=0

		return context,h,c,cls_label_pred
		
class lstm_target(nn.Module):
	
	def __init__(self,layer,dim,dropout,vocab_num,n_cls):
		super(lstm_target, self).__init__()
		self.n_cls=n_cls
		if self.n_cls:
			self.cls_embedding=nn.Embedding(n_cls,dim)
			self.lstmt=nn.LSTM(dim*3,dim,num_layers=layer,batch_first=True,bias=False,dropout=dropout)
		else:
			self.lstmt=nn.LSTM(dim*2,dim,num_layers=layer,batch_first=True,bias=False,dropout=dropout)
		self.atten_feed=attention_feed(dim)
		self.soft_atten=softattention(dim)
		self.softlinear=nn.Linear(dim,vocab_num,False)
		
	def forward(self,context,h,c,embedding,cls_label):
		context1=self.atten_feed(h[-1],context)
		lstm_input=torch.cat((embedding,context1),-1)
		if self.n_cls:
			cls_embed=self.cls_embedding(cls_label)
			lstm_input=torch.cat((lstm_input,cls_embed),-1)
		_,(h,c)=self.lstmt(lstm_input.unsqueeze(1),(h,c))
		pred=self.soft_atten(h[-1],context)
		pred=self.softlinear(pred)
		return pred,h,c


class lstm(nn.Module):

	def __init__(self,embed=None,device='cuda',layer=2,dim=1024,dropout=0.2,vocab_num=50000,n_cls=0):
		super(lstm, self).__init__()
		self.device = device
		self.encoder=lstm_source(layer,dim,dropout,n_cls)
		self.decoder=lstm_target(layer,dim,dropout,vocab_num+4,n_cls)
		self.embed=nn.Embedding(vocab_num+4,dim,padding_idx=0)
		self.embed.weight.data[1:].uniform_(-0.1,0.1)
		if embed!=None:
			self.embed=nn.Embedding(vocab_num+4,embed.weight.data.size(1),padding_idx=0)
			self.embed.weight.data[1:].uniform_(-0.1,0.1)
			print("Using external embedding file")
			self.embed.weight.data[3:-1]=embed.weight.data
			if embed.weight.data.size(1)!=dim:
				self.dense=nn.Linear(embed.weight.data.size(1),dim,bias=False)
		w=torch.ones(vocab_num+4).to(device)
		w[:2]=0
		w[-1]=0
		self.loss_function=torch.nn.CrossEntropyLoss(w,ignore_index=0, reduction='sum')

	def forward(self,source,mask_s,target,mask_t,length,cls_label):
		try:
			source_embed=self.dense(self.embed(source))
		except AttributeError:
			source_embed=self.embed(source)
		context,h,c,cls_label_pred=self.encoder(source_embed,length,cls_label)
		cls_label = cls_label_pred.argmax(1)
		loss=0
		test=0
		for i in range(target.size(1)-1):
			try:
				target_embed=self.dense(self.embed(target[:,i]))
			except AttributeError:
				target_embed=self.embed(target[:,i])
			if i==0:
				pred,h,c=self.decoder(context,h,c,target_embed,cls_label)
				predicted_word = torch.argmax(pred,1).unsqueeze(1)
				prediction = torch.argmax(pred,1).unsqueeze(1)
			else:
				pred,h,c=self.decoder(context,h,c,self.embed(predicted_word).squeeze(1),cls_label)
				predicted_word = torch.argmax(pred,1).unsqueeze(1)
				prediction = torch.cat((prediction,predicted_word.clone()),1)
			loss+=self.loss_function(pred,target[:,i+1])

		return prediction,loss


class lstmdecoder():

	def __init__(self, args):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if args.cpu:
			self.device="cpu"
		self.args=args
		# args
		
		self.cls_train=args.cls_train
		self.output_folder=args.output_folder
		self.output=[path.join(self.output_folder,args.model_folder.split('/')[-1])+'_output_'+test for test in args.test_set.split(',')]
		self.log=path.join(self.output_folder,'new_log.txt')
		self.vocab=path.join(args.data_folder,"vocab")
		self.dec=decoder(self.vocab)

		self.batch_size = args.batch_size 
		self.dim=args.dimension
		self.init_weight=args.init_weight

		self.test_path=[path.join(args.data_folder,test) for test in args.test_set.split(",")]
		self.dataset_size=[]
		for test in self.test_path:
			data_size=0
			with open(test,'r') as text:
				for line in text:
					data_size+=1
			self.dataset_size.append(data_size)

		# prepare for batch
		enc=encoder(self.vocab)
		self.data=data(args.batch_size, enc, args.max_length)
		
		if self.cls_train:
			self.model=lstm(None,self.device,args.layer,self.dim,dropout=0.2,vocab_num=50000,n_cls=len(self.test_path))
		else:
			self.model=lstm(None,self.device,args.layer,self.dim,dropout=0.2,vocab_num=50000,n_cls=0)

		self.model.encoder.apply(self.weights_init)
		self.model.decoder.apply(self.weights_init)
		self.model.to(self.device)

		self.model.load_state_dict(torch.load(args.model))
		print("read model done")

		with open(self.log,'w') as log:
			log.write("")

		for o in self.output:
			with open(o,'w') as output:
				output.write("")



	def weights_init(self,module):
		classname=module.__class__.__name__
		try:
			module.weight.data.uniform_(-self.init_weight,self.init_weight)
		except:
			pass

	def write_log(self, test_perp):
		with open(self.log, 'a') as log:
			# log.write("Iter {}\n".format(self.iter))
			# if self.iter!=0:
			# 	try:
			# 		log.write("Training Perp: {}\n".format(self.training_perp))
			# 	except:
			# 		pass
			# if len(test_perp)==1:
			# 	log.write("[Average]\n")
			# 	log.write("Test Perp: {}\n".format(test_perp[-1]))
			# else:
			for i in range(len(self.test_path)):
				log.write("[{}]\n".format(self.test_path[i]))
				log.write("Test Perp: {}\n".format(test_perp[i]))
			log.write("[Average]\n")
			log.write("Test Perp: {}\n".format(test_perp[-1]))
			log.write("\n")


	def decode(self):
		self.model.eval()
		total_loss_list=[]
		total_tokens_list=[]
		perp_list=[]
		total_num=[x//self.batch_size+1 for x in self.dataset_size]
		for test_n in range(len(self.test_path)):
			if self.cls_train:
				cls_label=torch.ones(self.batch_size).long()*test_n
				cls_label=cls_label.to(self.device)
			else:
				cls_label1=None
			t_num=total_num[test_n]
			test=self.test_path[test_n]
			total_loss=0
			total_tokens=0
			num=0
			print()
			while True:
				END,sources,targets,_,mask_s,mask_t,length,token_num,origin = self.data.train_batch(test, num)
				if sources is None:
					break
				sources=sources.to(self.device)
				targets=targets.to(self.device)
				mask_s=mask_s.to(self.device)
				mask_t=mask_t.to(self.device)
				length=length.to(self.device)
				try:
					cls_label1=cls_label[:sources.size(0)]
				except:
					pass
				total_tokens+=token_num

				with torch.no_grad():
					pred,loss=self.model(sources,mask_s,targets,mask_t,length,cls_label1)
					total_loss+=loss.item()
					for i in range(pred.size(0)):
						source = origin[i]
						target = self.dec.decode(pred[i].cpu().numpy())
						with open(self.output[test_n],'a') as output:
							output.write(source)
							output.write("|")
							output.write(target)
							output.write("\n")
				if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
					print("PROGRESS... "+str((num*self.batch_size*100+sum(self.dataset_size[:test_n])*100)//sum(self.dataset_size))+"%")
				if END==1:
					break
				num+=1
			print()
			print(total_loss/total_tokens)
			# perp=str((1/math.exp(-total_loss/total_tokens)))
			# total_loss_list.append(total_loss)
			# total_tokens_list.append(total_tokens)
			# perp_list.append(perp)
			# print("Test Perp for {}: {}".format(test.split("/")[-1], perp))
		# self.current=float(perp_list[-1])
		# if len(perp_list)>1:
		# 	perp=str((1/math.exp(-sum(total_loss_list)/sum(total_tokens_list))))
		# 	perp_list.append(perp)
		# 	print("Average Test Perp: {}".format(perp))
		# self.write_log(perp_list)