# Instruction for running

Before train/decode on GPT, please install the provided version of pre-trained library named "pytorch_pretrained". Some changes were directly made in the file "modeling_gpt2.py".


## Training data

The training data is open-source and can be found in the Internet. Otherwise, you could download from [here](https://drive.google.com/file/d/1HnjVjInXbnOErUn8NE0kKIR9VVXR0OTO/view?usp=sharing).


## DF

python generate_df.py --(arg) (arg value)

For alphaDF, use --alpha.
For LSTM model, use --LSTM.
For GPT model, use --GPT.
To write a new DF file, use --write.

## Train

python trainGPT.py --(arg) (arg value)
python trainlstm.py --(arg) (arg value)

For multi-task labeled and weighted learning, use --method "mtl" and --method "weight"

## Decode

python decodeGPT.py --(arg) (arg value)
python decodelstm.py --(arg) (arg value)

For multi-task labeled and weighted learning, use --method "mtl" and --method "weight"