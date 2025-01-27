import os
import torch

import numpy as np

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# bert embedding
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
from transformers.models.bert.configuration_bert import BertConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def bert_embedding(vecs, model_path):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config).to(device)

    vecs = tokenizer(vecs, padding="longest", return_tensors="pt")

    embedding = []

    for x, y in zip(vecs["input_ids"], vecs["attention_mask"]):
        hidden = model(x.unsqueeze(0).to(device), attention_mask = y.unsqueeze(0).to(device))[0]
        hidden = hidden.sum(axis=1) / y.sum(axis=-1).to(device)

        embedding.extend(hidden.cpu().data.numpy())

    return np.array(embedding)