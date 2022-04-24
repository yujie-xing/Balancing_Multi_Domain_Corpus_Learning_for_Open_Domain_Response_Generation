from pytorch_pretrained_bert import GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2Config, OpenAIAdam, BertAdam, GPT2Model
import torch
from torch import tensor
import pickle
import linecache
import numpy as np
from os import path
import math


class GPTdata():

	def __init__(self, batch_size, test_batch_size, max_length=100, end_id=50256):
		self.batch_size=batch_size
		self.test_batch_size=test_batch_size
		self.max_length=max_length
		self.length=2*max_length+3
		self.tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
		self.end=end_id
		self.pad=end_id+1
		self.sep=end_id+2

	def train_batch(self, file, num):
		sources=np.ones((self.batch_size, self.length))*self.pad
		targets=-np.ones((self.batch_size, self.length))
		positions=np.zeros((self.batch_size, self.length))
		cls_label=-np.ones(self.batch_size)
		tokens=np.zeros((self.batch_size, self.length))
		tokens[:, self.max_length:]=1
		l_s_set=set()
		l_t_set=set()
		END=0
		for i in range(self.batch_size):
			line=linecache.getline(file,num*self.batch_size+i+1).strip().split("|")
			if line==[""]:
				END=1
				break
			s=" ".join(line[-2].split())
			t=" ".join(line[-1].split())
			source = self.tokenizer.encode(s)[:self.max_length]
			target = [self.sep]+self.tokenizer.encode(t)[:self.max_length]+[self.end]
			l_s=len(source)
			l_t=len(target)-1
			l_s_set.add(l_s)
			l_t_set.add(l_t)
			sources[i, self.max_length-l_s:self.max_length]=source
			sources[i, self.max_length : self.max_length+l_t]=target[:-1]
			targets[i, self.max_length : self.max_length+l_t]=target[1:]
			positions[i, self.max_length-l_s:]=range(sources.shape[1]-self.max_length+l_s)
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
		sources=sources[:i, self.max_length-max_l_s : self.max_length+max_l_t]
		targets=targets[:i, self.max_length-max_l_s : self.max_length+max_l_t]
		positions=positions[:i, self.max_length-max_l_s : self.max_length+max_l_t]
		cls_label=cls_label[:i]
		tokens=tokens[:i, self.max_length-max_l_s : self.max_length+max_l_t]
		mask=np.ones(sources.shape)*(sources!=self.pad)
		token_num=(targets!=-1).sum()

		return END,tensor(sources).long(),tensor(targets).long(),tensor(positions).long(),tensor(cls_label).long(),tensor(tokens).long(),tensor(mask).float(),token_num,max_l_s



	def test_batch(self, file, num, cls_train):
		origin=[]
		sources=np.ones((self.test_batch_size, self.max_length+1))*self.pad
		targets=-np.ones((self.test_batch_size, self.max_length))
		positions=np.zeros((self.test_batch_size, self.max_length+1))
		tokens=np.zeros((self.test_batch_size, self.max_length+1))
		tokens[:,-1]=1
		l_s_set=set()
		l_t_set=set()
		END=0
		for i in range(self.test_batch_size):
			line=linecache.getline(file,num*self.test_batch_size+i+1).strip().split("|")
			if line==[""]:
				END=1
				break
			s=" ".join(line[-2].split())
			source = self.tokenizer.encode(s)[:self.max_length]
			try: 
				t=" ".join(line[-1].split())
				target = self.tokenizer.encode(t)[:self.max_length-1]+[self.end]
			except IndexError:
				target = [self.end]
			l_s=len(source)
			l_t=len(target)
			l_s_set.add(l_s)
			l_t_set.add(l_t)
			origin.append(line[-2])
			sources[i, -l_s-1:-1]=source
			sources[i, -1]=self.sep
			targets[i, :l_t]=target
			positions[i, -l_s-1:]=range(l_s+1)
		try:
			max_l_s=max(l_s_set)
			max_l_t=max(l_t_set)
		except ValueError:
			return END,None,None,None,None,None,None,None
		sources=sources[:i, -max_l_s-1:]
		targets=targets[:i, :max_l_t]
		positions=positions[:i, -max_l_s-1:]
		tokens=tokens[:i, -max_l_s-1:]
		mask=np.ones(sources.shape)*(sources!=self.pad)
		token_num=(targets!=-1).sum()
		return END,tensor(sources).long(),tensor(targets).long(),tensor(positions).long(),tensor(tokens).long(),tensor(mask).float(),token_num,origin

	def decode(self,tokens):
		return self.tokenizer.decode(tokens, exclude=[self.sep, self.pad])


class GPT():

	def __init__(self, args):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if args.cpu:
			self.device="cpu"
		# args
		self.max_iter=args.max_iter
		self.end_id=args.end_id
		self.dimension=args.dimension
		self.save_folder=args.save_folder
		self.save_model=not args.no_save
		self.log=path.join(self.save_folder,"log.txt")
		self.cls_train=args.cls_train
		self.ordered=args.ordered
		self.batch_size=args.batch_size
		self.test_batch_size=args.test_batch_size
		self.fine_tune_model=path.join(args.fine_tune_folder,args.fine_tune_model)
		self.fine_tune_turn=args.fine_tune_turn

		# prepare for the model
		if args.fine_tune:
			with open(path.join(args.fine_tune_folder,'config'),'rb') as c:
				config=pickle.load(c)
			model_state_dict = torch.load(self.fine_tune_model)
			self.model = GPT2DoubleHeadsModel(config)
			self.model.load_state_dict(model_state_dict)
			self.model.to(self.device)

		# training path & dataset size
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

		self.n_special=args.n_special
		self.token_num=args.n_token
		if self.cls_train:
			self.n_token=args.n_token+len(self.valid_path)
		else:
			self.n_token=args.n_token
		# prepare for batch
		self.data=GPTdata(args.batch_size,
						args.test_batch_size,
						args.max_length,
						args.end_id)
		
		# prepare for the model
		if args.no_gpt:
			config=GPT2Config(vocab_size_or_config_json_file=args.end_id+1,
							n_special=self.n_special,
							n_token=self.n_token,
							n_cls=len(self.valid_path),
							n_positions=2*args.max_length+3,
							n_ctx=args.dimension,
							n_embd=args.dimension,
							n_layer=args.layer,
							n_head=args.head,
							layer_norm_epsilon=args.layer_norm_epsilon,
							initializer_range=args.initial_std)
			self.model=GPT2DoubleHeadsModel(config)
			# model_state_dict = torch.load(args.model_name_or_path)
			# self.model.load_state_dict(model_state_dict)
			self.model.to(self.device)
		elif args.fine_tune and self.cls_train:
			with open(path.join(args.fine_tune_folder,'config'),'rb') as c:
				config=pickle.load(c)
			config.n_cls=len(self.valid_path)
			config.n_token=self.n_token
			self.model = GPT2DoubleHeadsModel(config)
			self.model.to(self.device)
			trained_model = torch.load(self.fine_tune_model)
			tte=trained_model["transformer.tte.weight"].data
			cls_linear=trained_model["cls_head.linear.weight"].data
			self.model.transformer.tte.weight.data[:tte.size(0)]=tte
			self.model.cls_head.linear.weight.data[:cls_linear.size(0)]=cls_linear
			trained_model["transformer.tte.weight"].data=self.model.transformer.tte.weight.data
			trained_model["cls_head.linear.weight"]=self.model.cls_head.linear.weight.data
			self.model.load_state_dict(trained_model)
		elif args.fine_tune:
			with open(path.join(args.fine_tune_folder,'config'),'rb') as c:
				config=pickle.load(c)
			model_state_dict = torch.load(self.fine_tune_model)
			self.model = GPT2DoubleHeadsModel(config)
			self.model.to(self.device)
			self.model.load_state_dict(model_state_dict)
		else:
			self.model=GPT2DoubleHeadsModel.from_pretrained(args.model_name_or_path, n_special=self.n_special, n_token=self.n_token, n_cls=len(self.valid_path))
			config=self.model.config
			self.model.to(self.device)
		# optimizer
		param_optimizer = list(self.model.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]
		num_train_optimization_steps = (sum([(dataset_size//args.batch_size)+1 for dataset_size in self.dataset_size])) * args.max_iter
		if args.optimizer=="GPT":
			self.optimizer=OpenAIAdam(optimizer_grouped_parameters,
									lr=args.lr,
									warmup=args.warmup_proportion,
									max_grad_norm=args.max_grad_norm,
									weight_decay=args.weight_decay,
									t_total=num_train_optimization_steps)
		elif args.optimizer=="Bert":
			self.optimizer = BertAdam(optimizer_grouped_parameters,
									 lr=args.lr,
									 warmup=args.warmup_proportion,
									 t_total=num_train_optimization_steps)
		
		# save parameters
		with open(path.join(self.save_folder,"config"),'wb') as c:
				pickle.dump(config,c)
		with open(path.join(self.save_folder,"params"),'wb') as parameters:
				pickle.dump(args,parameters)
		with open(self.log,'w') as log:
			log.write(" ".join(self.train_path)+"\n\n")
			# log.write(args)
			log.write("\n\n")


	def save(self):
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		torch.save(model_to_save.state_dict(), path.join(self.save_folder, "model"+str(self.iter)))
	def write_log(self, test_perp):
		with open(self.log, 'a') as log:
			log.write("Iter {}\n".format(self.iter))
			if self.iter!=0:
				log.write("Training Perp: {}\n".format(self.training_perp))
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

	def train(self):
		self.iter=self.fine_tune_turn
		print("Iter {}".format(self.iter))
		self.test()
		total_num=[x//self.batch_size+1 for x in self.dataset_size]
		while True:
			self.model.train()
			self.iter+=1
			if self.iter>self.max_iter:
				break
			print("\nIter {}".format(self.iter))
			total_loss=0
			total_tokens=0
			for train_n in range(len(self.train_path)):
				if self.cls_train:
					cls_label2=torch.ones(self.batch_size).long()*train_n
					cls_label2=cls_label2.to(self.device)
				else:
					cls_label1=None
				t_num=total_num[train_n]
				train=self.train_path[train_n]
				num=0
				print()
				while True:
					self.optimizer.zero_grad()
					END,sources,targets,positions,cls_label,tokens,mask,token_num,max_l_s= self.data.train_batch(train, num)
					if sources is None:
						break
					sources=sources.to(self.device)
					targets=targets.to(self.device)
					positions=positions.to(self.device)
					cls_label=cls_label.to(self.device)
					tokens=tokens.to(self.device)
					mask=mask.to(self.device)
					mask=torch.matmul(mask.unsqueeze(2),mask.unsqueeze(1))
					mask=mask.byte()
					triu=np.triu(np.ones((sources.shape[0],sources.shape[1]-max_l_s-1,sources.shape[1]-max_l_s-1)),k=1)
					triu=tensor(triu).float()
					triu=torch.cat((torch.ones(sources.shape[0],max_l_s+1,sources.shape[1]-max_l_s-1),triu),1)
					triu=torch.cat((torch.zeros(sources.shape[0],sources.shape[1],max_l_s+1),triu),2)
					triu=(triu==0).byte().to(self.device)
					mask=mask&triu
					mask=mask.unsqueeze(1)
					try:
						cls_label1=cls_label2[:sources.size(0)]
					except:
						pass
					total_tokens+=token_num
					if self.cls_train and self.ordered:
						tokens[:,max_l_s:]=cls_label1.unsqueeze(-1)+self.token_num
					elif self.cls_train:
						tokens[:,max_l_s:]=cls_label.unsqueeze(-1)+self.token_num
					loss=self.model(sources, positions, tokens, targets, max_l_s, None, None, mask)
					loss=sum(loss)
					loss.backward()
					with torch.no_grad():
						total_loss+=loss.item()
					self.optimizer.step()
					if num in {t_num//10,t_num//5,t_num//3,t_num//2,t_num//1.5,t_num//1.25,t_num//1.1}:
						print("PROGRESS... "+str((num*self.batch_size*100+sum(self.dataset_size[:train_n])*100)//sum(self.dataset_size))+"%")
					num+=1
					if END==1:
						break
			self.training_perp=str((1/math.exp(-total_loss/total_tokens)))
			print("\nTraining Perp: {}\n".format(self.training_perp))
			self.test()
			if self.save_model:
				self.save()

	def test(self):
		self.model.eval()
		total_loss_list=[]
		total_tokens_list=[]
		perp_list=[]
		total_num=[x//self.test_batch_size+1 for x in self.valid_size]
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
				END,sources,targets,positions,_,tokens,mask,token_num,max_l_s = self.data.train_batch(valid, num)
				if sources is None:
					break
				sources=sources.to(self.device)
				targets=targets.to(self.device)
				positions=positions.to(self.device)
				tokens=tokens.to(self.device)
				mask=mask.to(self.device)
				mask=torch.matmul(mask.unsqueeze(2),mask.unsqueeze(1))
				mask=mask.byte()
				triu=np.triu(np.ones((sources.shape[0],sources.shape[1]-max_l_s-1,sources.shape[1]-max_l_s-1)),k=1)
				triu=tensor(triu).float()
				triu=torch.cat((torch.ones(sources.shape[0],max_l_s+1,sources.shape[1]-max_l_s-1),triu),1)
				triu=torch.cat((torch.zeros(sources.shape[0],sources.shape[1],max_l_s+1),triu),2)
				triu=(triu==0).byte().to(self.device)
				mask=mask&triu
				mask=mask.unsqueeze(1)
				total_tokens+=token_num
				try:
					cls_label1=cls_label[:sources.size(0)]
				except:
					pass
				with torch.no_grad():
					if self.cls_train:
						tokens[:,max_l_s:]=cls_label1.unsqueeze(-1)+self.token_num
					loss=self.model(sources, positions, tokens, targets, max_l_s, None, None, mask)
					total_loss+=loss[0].item()
				if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
					print("PROGRESS... "+str((num*self.test_batch_size*100+sum(self.valid_size[:valid_n])*100)//sum(self.valid_size))+"%")
				if END==1:
					break
				num+=1
			print()
			# print(total_loss/total_tokens)
			perp=str((1/math.exp(-total_loss/total_tokens)))
			total_loss_list.append(total_loss)
			total_tokens_list.append(total_tokens)
			perp_list.append(perp)
			print("Test Perp for {}: {}".format(valid.split("/")[-1], perp))
		if len(perp_list)>1:
			perp=str((1/math.exp(-sum(total_loss_list)/sum(total_tokens_list))))
			perp_list.append(perp)
			print("\nAverage Test Perp: {}".format(perp))
		self.write_log(perp_list)




class GPTdecoder():

	def __init__(self, args):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if args.cpu:
			self.device="cpu"
		self.model_path=path.join(args.model_folder,args.model_name)
		self.model_folder=args.model_folder
		self.test_path=[path.join(args.data_folder,test) for test in args.test_set.split(",")]
		self.batch_size=args.batch_size
		self.max_length=args.max_length
		self.cls_train=args.cls_train
		# read config
		with open(path.join(self.model_folder,'config'),'rb') as c:
			config=pickle.load(c)
		# read parameter
		with open(path.join(self.model_folder,'params'),'rb') as parameter:
			train_args=pickle.load(parameter)
		# prepare for batch
		self.end_id=train_args.end_id
		self.data=GPTdata(self.batch_size,
						self.batch_size,
						args.max_length,
						self.end_id)
		
		# prepare for the model
		model_state_dict = torch.load(self.model_path)
		# for key in model_state_dict:
			# model_state_dict[key]=model_state_dict[key].to(self.device)
		self.model = GPT2DoubleHeadsModel(config)
		self.model.to(self.device)
		self.model.load_state_dict(model_state_dict)
		
		self.dataset_size=[]
		for test in self.test_path:
			data_size=0
			with open(test,'r') as text:
				for line in text:
					data_size+=1
			self.dataset_size.append(data_size)
		# prepare for the output
		self.log=path.join(train_args.save_folder,"new_log.txt")
		with open(self.log,'w') as log:
			log.write("")
		self.output=[path.join(args.output_folder,args.model_folder.split("/")[-1])+"_output_"+test for test in args.test_set.split(",")]
		for o in self.output:
			with open(o,'w') as output:
				output.write("")


	def write_log(self, test_perp):
		with open(self.log, 'a') as log:
			if len(test_perp)==1:
				log.write("[Average]\n")
				log.write("Test Perp: {}\n".format(test_perp[-1]))
			else:
				for i in range(len(self.test_path)):
					log.write("[{}]\n".format(self.test_path[i]))
					log.write("Test Perp: {}\n".format(test_perp[i]))
			log.write("\n")

	def test(self):
		self.model.eval()
		total_loss_list=[]
		total_tokens_list=[]
		perp_list=[]
		total_num=[x//self.batch_size+1 for x in self.dataset_size]
		for test_n in range(len(self.test_path)):
			t_num=total_num[test_n]
			test=self.test_path[test_n]
			total_loss=0
			total_tokens=0
			num=0
			while True:
				END,sources,targets,positions,tokens,mask,token_num,max_l_s = self.data.train_batch(test, num)
				if sources is None:
					break
				sources=sources.to(self.device)
				targets=targets.to(self.device)
				positions=positions.to(self.device)
				tokens=tokens.to(self.device)
				mask=mask.to(self.device)
				total_tokens+=token_num
				with torch.no_grad():
					loss=self.model(sources, positions, tokens, targets, max_l_s, None, None, mask)				
					total_loss+=loss[0].item()
				if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
					print("PROGRESS... "+str((num*self.batch_size*100+sum(self.dataset_size[:valid_n])*100)//sum(self.dataset_size))+"%")
				if END==1:
					break
				num+=1
			perp=str((1/math.exp(-total_loss/total_tokens)))
			total_loss_list.append(total_loss)
			total_tokens_list.append(total_tokens)
			perp_list.append(perp)
			print("Test Perp for {}: {}".format(test.split("/")[-1], perp))
		if len(perp_list)>1:
			perp=str((1/math.exp(-sum(total_loss_list)/sum(total_tokens_list))))
			perp_list.append(perp)
			print("Average Test Perp: {}".format(perp))
		self.write_log(perp_list)

	def decode(self):
		self.model.eval()
		total_num=[x//self.batch_size+1 for x in self.dataset_size]
		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
		for test_n in range(len(self.test_path)):
			test=self.test_path[test_n]
			t_num=total_num[test_n]
			total_loss=0
			total_tokens=0
			num=0
			while True:
				END,sources,targets,positions,tokens,mask,token_num,origin = self.data.test_batch(test, num, self.cls_train)
				if sources is None:
					break
				sources=sources.to(self.device)
				s=sources[:]
				targets=targets.to(self.device)
				positions=positions.to(self.device)
				tokens=tokens.to(self.device)
				if self.cls_train:
					tokens=tokens*(test_n+2)
				mask=mask.to(self.device)
				mask1=torch.matmul(mask.unsqueeze(2),mask.unsqueeze(1)).unsqueeze(1).byte()		
				total_tokens+=token_num
				end=torch.ones(sources.size(0))*self.end_id
				end=end.to(self.device).long()
				with torch.no_grad():

					if self.cls_train:
					
						_,cls_logits,_=self.model(sources, positions, tokens, None, sources.size(1)-1, None, None, mask1)
						cls_pred = torch.argmax(cls_logits,1)
						tokens[:,-1]=cls_pred+2

					lm_logit, _=self.model(sources, positions, tokens, None, None, None, None, mask1)
					lm_logit=lm_logit[:,-1,:]
					predicted_word=torch.argmax(lm_logit,1).unsqueeze(1)
					prediction=predicted_word.clone()
					predicted_lm_logit=lm_logit.clone().unsqueeze(1)
					for word in range(self.max_length):
						sources=torch.cat((sources,predicted_word),1)
						positions=torch.cat((positions,(positions[:,-1]+1).unsqueeze(1)),1)
						tokens=torch.cat((tokens,tokens[:,-1].unsqueeze(1)),1)
						mask=torch.cat((mask,mask[:,-1].unsqueeze(1)),1)
						mask1=torch.matmul(mask.unsqueeze(2),mask.unsqueeze(1)).unsqueeze(1).byte()		
						
						lm_logit, _=self.model(sources, positions, tokens, None, None, None, None, mask1)
						lm_logit=lm_logit[:,-1,:]
						predicted_word=torch.argmax(lm_logit,1).unsqueeze(1)
						prediction=torch.cat((prediction,predicted_word.clone()),1)
						predicted_lm_logit=torch.cat((predicted_lm_logit,lm_logit.clone().unsqueeze(1)),1)
						if torch.all(torch.eq(predicted_word.squeeze(-1),end)):
							break
					targets=targets[:,:prediction.size(1)]
					try:
						loss = loss_fct(predicted_lm_logit.view(-1, predicted_lm_logit.size(-1)), targets.view(-1)).item()
					except ValueError:
						loss = 0
					total_loss+=loss
					for i in range(prediction.size(0)):
						source=origin[i]
						p=prediction[i].cpu().numpy()
						try:
							target = self.data.decode(p[:np.where(p==self.end_id)[0][0]])
						except IndexError:
							target = self.data.decode(p)
						target_word_list = target.split()
						try:
							target = [target_word_list[0]]
						except:
							target = ""
							print(1)
						if len(target_word_list)>1:
							for i in range(1,len(target_word_list)):
								current_word = target_word_list[i]
								previous_word = target_word_list[i-1]
								if current_word != previous_word:
									target.append(current_word)
								else:
									break
						target = " ".join(target)
						# print(source+"|"+target)
						with open(self.output[test_n],'a') as output:
							output.write(source)
							output.write("|")
							output.write(target+"\n")
				# Track progress:
				# if num in {t_num//5,t_num//2,t_num//1.25,t_num//1.1}:
					# print(str((num*self.batch_size*100+sum(self.dataset_size[:test_n])*100)//sum(self.dataset_size))+"%")
				if END==1:
					break
				num+=1
			# Write log:
			# perp=str((1/math.exp(-total_loss/total_tokens)))
			# total_loss_list.append(total_loss)
			# total_tokens_list.append(total_tokens)
			# perp_list.append(perp)
			# print("\nTest Perp for {}: {}".format(test.split("/")[-1], perp))
		# if len(perp_list)>1:
			# perp=str((1/math.exp(-sum(total_loss_list)/sum(total_tokens_list))))
			# perp_list.append(perp)
			# print("Average Test Perp: {}".format(perp))
		# self.write_log(perp_list)
