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

class data():

	def __init__(self, batch_size, encoder, max_length=50):
		self.batch_size=batch_size
		self.max_length=max_length
		self.tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')
		self.encoder=encoder

	def train_batch(self, file, num, df):
		sources=np.zeros((self.batch_size, self.max_length+2))
		targets=np.zeros((self.batch_size, self.max_length+2))
		cls_label=-np.ones(self.batch_size)
		weight = np.ones((self.batch_size,self.max_length+2))
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
			sources[i, :l_s]=source
			targets[i, :l_t]=target
			try:
				cls_label[i]=int(line[0])
				weight[i, 1:l_t-1]=[df[int(line[0])][word] for word in target[1:-1]]
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
		weight = weight[:i, : max_l_t]
		mask_s=np.ones(sources.shape)*(sources!=0)
		mask_t=np.ones(targets.shape)*(targets!=0)
		length_s=(sources!=0).sum(1)
		token_num=mask_t[:,1:].sum()

		return END,tensor(sources).long(),tensor(targets).long(),tensor(cls_label).long(),tensor(mask_s).long(),tensor(mask_t).long(),tensor(length_s).long(),token_num,tensor(weight).float()
	
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
		self.loss_function=torch.nn.CrossEntropyLoss(w,ignore_index=0, reduction='none')
		self.cls_loss_function=torch.nn.CrossEntropyLoss(reduction='sum')
		
	def forward(self,source,mask_s,target,mask_t,length,cls_label,cls_train=0,weight=None):
		try:
			source_embed=self.dense(self.embed(source))
		except AttributeError:
			source_embed=self.embed(source)
		context,h,c,cls_label_pred=self.encoder(source_embed,length,cls_label)
		if cls_label is not None:
			cls_loss = self.cls_loss_function(cls_label_pred,cls_label)
			cls_label = cls_label_pred.argmax(1)
		loss=0
		for i in range(target.size(1)-1):
			try:
				target_embed=self.dense(self.embed(target[:,i]))
			except AttributeError:
				target_embed=self.embed(target[:,i])
			pred,h,c=self.decoder(context,h,c,target_embed,cls_label)
			loss0 = self.loss_function(pred,target[:,i+1])
			if weight is not None:
				loss0 *= weight[:,i+1]
			loss+=loss0.sum()
		if cls_train:
			token_num = mask_t[:,1:].sum()
			loss+=cls_loss

		return loss



class seq2seq():

	def __init__(self, args):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if args.cpu:
			self.device="cpu"
		self.args=args
		# args
		self.max_iter=args.max_iter
		self.dim=args.dimension
		self.save_folder=args.save_folder
		self.save_model=not args.no_save
		self.log=path.join(self.save_folder,"lstmlog.txt")
		self.cls_train=args.cls_train
		self.ordered=args.ordered
		self.batch_size=args.batch_size
		self.init_weight=args.init_weight
		self.lr=args.lr
		self.start_half=args.start_half
		self.thred=args.max_grad_norm
		self.optim_type=args.optim
		self.embedding=args.embedding
		self.vocab=path.join(args.data_folder,"vocab")
		self.fine_tune=args.fine_tune
		self.fine_tune_turn=args.fine_tune_turn

		# training path & dataset size
		with open(path.join(args.data_folder,args.df_file),'rb') as P:
			self.df = pickle.load(P)
		self.train_path=[path.join(args.data_folder,train) for train in args.train_set.split(",")]
		self.valid_path=[path.join(args.data_folder,valid) for valid in args.valid_set.split(",")]
		self.dataset_size=[]
		self.valid_size=[]
		for train in self.train_path:
			data_size=0
			with open(train,'r') as text:
				for line in text:
					data_size+=1
			self.dataset_size.append(data_size)
		for valid in self.valid_path:
			data_size=0
			with open(valid,'r') as text:
				for line in text:
					data_size+=1
			self.valid_size.append(data_size)

		# prepare for the embedding
		try:
			embed=torch.load(self.embedding)
		except FileNotFoundError:
			embed=None

		# prepare for batch
		enc=encoder(self.vocab)
		self.data=data(args.batch_size, enc, args.max_length)
		
		if self.cls_train:
			self.model=lstm(embed,self.device,args.layer,self.dim,dropout=0.2,vocab_num=50000,n_cls=len(self.valid_path))
		else:
			self.model=lstm(embed,self.device,args.layer,self.dim,dropout=0.2,vocab_num=50000,n_cls=0)

		self.model.encoder.apply(self.weights_init)
		self.model.decoder.apply(self.weights_init)
		self.model.to(self.device)

		if self.fine_tune and self.cls_train:
			loaded_model = torch.load(args.fine_tune_path)
			for name in self.model.state_dict():
				if name not in loaded_model:
					loaded_model[name]=self.model.state_dict()[name]
			lstmt_embedding = loaded_model["decoder.lstmt.weight_ih_l0"].data
			origin = self.model.state_dict()["decoder.lstmt.weight_ih_l0"]
			origin[:,:lstmt_embedding.size(1)] = lstmt_embedding
			loaded_model["decoder.lstmt.weight_ih_l0"].data = origin
			self.model.load_state_dict(loaded_model)
			print("read model done")
		elif self.fine_tune:
			self.model.load_state_dict(torch.load(args.fine_tune_path))
			print("read model done")

		# optimizer
		num_train_optimization_steps = (sum([(dataset_size//args.batch_size)+1 for dataset_size in self.dataset_size])) * args.max_iter
		if self.optim_type=='adam':
			self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
		
		save parameters
		if not self.fine_tune:
		with open(path.join(self.save_folder,"lstmparams"),'wb') as parameters:
				pickle.dump(args,parameters)
		with open(self.log,'w') as log:
			log.write(" ".join(self.train_path)+"\n\n")
			log.write(args.fine_tune_path)
			log.write("\n\n")

	def weights_init(self,module):
		classname=module.__class__.__name__
		try:
			module.weight.data.uniform_(-self.init_weight,self.init_weight)
		except:
			pass

	def save(self):
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		torch.save(model_to_save.state_dict(), path.join(self.save_folder, "lstmmodel"+str(self.iter)))
	def write_log(self, test_perp):
		with open(self.log, 'a') as log:
			log.write("Iter {}\n".format(self.iter))
			if self.iter!=0:
				try:
					log.write("Training Perp: {}\n".format(self.training_perp))
				except:
					pass
			if len(test_perp)==1:
				log.write("[Average]\n")
				log.write("Test Perp: {}\n".format(test_perp[-1]))
			else:
				for i in range(len(self.valid_path)):
					log.write("[{}]\n".format(self.valid_path[i]))
					log.write("Test Perp: {}\n".format(test_perp[i]))
				log.write("[Average]\n")
				log.write("Test Perp: {}\n".format(test_perp[-1]))
			log.write("\n")

	def update(self):
		lr=self.lr
		grad_norm=0
		for m in list(self.model.parameters()):
			m.grad.data = m.grad.data*(1/self.source_size)
			grad_norm+=m.grad.data.norm()**2
		grad_norm=grad_norm**0.5
		if grad_norm>self.thred:
			lr=lr*self.thred/grad_norm
		for f in self.model.parameters():
			f.data.sub_(f.grad.data * lr)

	def train(self):
		self.iter=self.fine_tune_turn
		print("Iter {}".format(self.iter))
		if self.fine_tune:
			self.test()
		total_num=[x//self.batch_size+1 for x in self.dataset_size]
		while True:
			self.model.train()
			self.iter+=1
			if self.iter>self.max_iter:
				break
			if self.iter>=self.start_half and self.start_half:
				self.lr*=0.5
			print("\nIter {}".format(self.iter))
			total_loss=0
			total_tokens=0
			for train_n in range(len(self.train_path)):
				if len(self.train_path)>1 and self.cls_train:
					cls_label2=torch.ones(self.batch_size).long()*train_n
					cls_label2=cls_label2.to(self.device)
				else:
					cls_label1=None
				t_num=total_num[train_n]
				train=self.train_path[train_n]
				num=0
				print()
				while True:
					self.model.zero_grad()
					END,sources,targets,cls_label,mask_s,mask_t,length,token_num,weight = self.data.train_batch(train, num, self.df)
					if sources is None:
						break
					sources=sources.to(self.device)
					targets=targets.to(self.device)
					cls_label=cls_label.to(self.device)
					weight=weight.to(self.device)
					mask_s=mask_s.to(self.device)
					mask_t=mask_t.to(self.device)
					length=length.to(self.device)
					self.source_size=sources.size(0)
					try:
						cls_label1=cls_label2[:sources.size(0)]
					except:
						pass

					total_tokens+=token_num

					if self.cls_train and self.ordered:
						loss=self.model(sources,mask_s,targets,mask_t,length,cls_label1)
					elif self.cls_train:
						loss=self.model(sources,mask_s,targets,mask_t,length,cls_label,cls_train=1)
					else: # Weighted Learning
						loss=self.model(sources,mask_s,targets,mask_t,length,None,cls_train=0,weight=weight)
					loss.backward()

					with torch.no_grad():
						total_loss+=loss.item()

					if self.optim_type=='sgd':
						self.update()
					else:
						self.optim.step()
		
					if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
						print("PROGRESS... "+str((num*self.batch_size*100+sum(self.dataset_size[:train_n])*100)//sum(self.dataset_size))+"%")
					num+=1
					if END==1:
						break
			self.training_perp=str((1/math.exp(-total_loss/total_tokens)))
			print("\nTraining Perp: {}\n".format(self.training_perp))
			self.test()
			if self.iter>=2 and self.save_model:
				self.save()

	def test(self):
		self.model.eval()
		total_loss_list=[]
		total_tokens_list=[]
		perp_list=[]
		total_num=[x//self.batch_size+1 for x in self.valid_size]
		for valid_n in range(len(self.valid_path)):
			if self.cls_train:
				cls_label=torch.ones(self.batch_size).long()*valid_n
				cls_label=cls_label.to(self.device)
			else:
				cls_label1=None
			t_num=total_num[valid_n]
			valid=self.valid_path[valid_n]
			total_loss=0
			total_tokens=0
			num=0
			print()
			while True:
				END,sources,targets,_,mask_s,mask_t,length,token_num,_ = self.data.train_batch(valid, num ,self.df)
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
					loss=self.model(sources,mask_s,targets,mask_t,length,cls_label1)
					total_loss+=loss.item()
				if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
					print("PROGRESS... "+str((num*self.batch_size*100+sum(self.valid_size[:valid_n])*100)//sum(self.valid_size))+"%")
				if END==1:
					break
				num+=1
			print()
			print(total_loss/total_tokens)
			perp=str((1/math.exp(-total_loss/total_tokens)))
			total_loss_list.append(total_loss)
			total_tokens_list.append(total_tokens)
			perp_list.append(perp)
			print("Test Perp for {}: {}".format(valid.split("/")[-1], perp))
		if len(perp_list)>1:
			perp=str((1/math.exp(-sum(total_loss_list)/sum(total_tokens_list))))
			perp_list.append(perp)
			print("Average Test Perp: {}".format(perp))
		self.write_log(perp_list)