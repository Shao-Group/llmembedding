import os
import time
import tqdm
import torch
import sys

import numpy as np

from utils import test_recall
from trainer import train_epoch
from datasets import TripletString, StringDataset
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open
from safetensors.torch import load_file

# bert embedding
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
from transformers.models.bert.configuration_bert import BertConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def _batch_embed(args, net, vecs, device):

    vecs = tokenizer(vecs, padding="longest", return_tensors="pt")

    #test_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False, num_workers=1)
    embedding = []

    for x, y in zip(vecs["input_ids"], vecs["attention_mask"]):
        #hidden = net(x.to(device))[0][:,:1,:]
        hidden = net(x.unsqueeze(0).to(device), attention_mask = y.unsqueeze(0).to(device))[0]
        hidden = hidden.sum(axis=1) / y.sum(axis=-1).to(device)

        embedding.extend(hidden.cpu().data.numpy())

    return np.array(embedding)


def bert_embedding(args, h, model_file):
    """
    h[DataHandler]
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader = TripletString(h.xt, h.nt, h.train_knn, h.train_dist, K=args.k)
    
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True, config=config).to(device)

    xt = _batch_embed(args, model, h.string_t, device)
    start_time = time.time()
    xt = []
    xb = _batch_embed(args, model, h.string_b, device)
    embed_time = time.time() - start_time
    xq = _batch_embed(args, model, h.string_q, device)
    print("# Embedding time: " + str(embed_time))
    if args.save_embed:
        if args.embed_dir != "":
            args.embed_dir = args.embed_dir + "/"
        os.makedirs("{}/{}".format(data_file, args.embed_dir), exist_ok=True)
        np.save("{}/{}embedding_xb".format(data_file, args.embed_dir), xb)
        np.save("{}/{}embedding_xt".format(data_file, args.embed_dir), xt)
        np.save("{}/{}embedding_xq".format(data_file, args.embed_dir), xq)

    if args.recall:
        test_recall(xb, xq, h.query_knn)
    return xq, xb, xt
