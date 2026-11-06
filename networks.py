import warnings
from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSeq2SeqLM,AutoModel,AutoModelForSequenceClassification
from transformers import logging as hf_logging
import logging
import os
import torch.nn as nn
# Set both loggers to ERROR level
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"#这行代码禁用了transformers库中的tokenization并行处理。如果你不设置这个环境变量，tokenization可能会在多个线程或进程中并行运行，这可能会引入随机性，尤其是在数据加载和处理的过程中。
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def get_tokenizer(model_name):
    #return AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name)
#AutoModel
def get_model(args):
    #config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    #config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)
    #return AutoModelForSequenceClassification.from_pretrained(args.model, config=config,trust_remote_code=True)
    return AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
def get_model1(args):
    config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, config=config,trust_remote_code=True)
    # 添加一个用于分类的额外线性层
    #model.classifier = nn.Linear(config.hidden_size, args.num_labels)
    return model
# def get_model(args):
#     return CustomModel(args.model, num_labels=args.num_labels)
#
#
# from transformers import AutoModelForSequenceClassification
# from tensorflow.keras.layers import Concatenate, Dense
# from tensorflow.keras.models import Model
#
# class CustomModel(Model):
#     def __init__(self, model_name_or_path, num_labels):
#         super().__init__()
#         self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
#         self.concat = Concatenate(axis=-1)
#
#     def call(self, inputs):
#         texts = {"input_ids": inputs["input_ids_text"], "attention_mask": inputs["attention_mask_text"], "token_type_ids": inputs["token_type_ids_text"]}
#         generated_texts = {"input_ids": inputs["input_ids_generated"], "attention_mask": inputs["attention_mask_generated"], "token_type_ids": inputs["token_type_ids_generated"]}
#         similarity_scores = inputs["similarity_scores"]
#         texts_feats = self.bert_model(texts)[0]  # 使用[0]获取模型输出的特征
#         generated_texts_feats = self.bert_model(generated_texts)[0]
#         x = self.concat([texts_feats, generated_texts_feats, similarity_scores])
#         return self.bert_model.classifier(x)  # 返回模型的分类器输出
