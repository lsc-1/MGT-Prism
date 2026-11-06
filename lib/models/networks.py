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

