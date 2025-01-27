### Introduction
This repository holds the code for a genomic large language model designed to produce sequence embeddings approximating the edit distance. It is trained via contrastive
learning based on a pretrained model. It also includes two experiments: correlation with edit distance and similar sequence searching.

This repository contains the code for a genomic large language model designed to produce sequence embeddings approximating the edit distance. The model is trained using contrastive learning techniques, leveraging a pretrained model as the foundation. It also includes two experiments:
- **Correlation with Edit Distance**
- **Similar Sequence Searching**

## Hugging Face Models

The pretrained models are available on [Hugging Face](https://huggingface.co) under the following repositories:
- [`PSUXL/LLMED-MAE`](https://huggingface.co/PSUXL/LLMED-MAE)
- [`PSUXL/LLMED-triplet`](https://huggingface.co/PSUXL/LLMED-triplet)
- [`PSUXL/LLMED-combined`](https://huggingface.co/PSUXL/LLMED-combined)

These models are based on the DNABERT2 pretrained model. Below is an example of how to use the models for sequence embeddings.

## Usage

Here is an example code snippet to generate embeddings using the PSUXL/LLMED-combined model:
```
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import torch
import numpy as np
import os

# Load DNABERT2 tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
from transformers.models.bert.configuration_bert import BertConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def bert_embedding(vecs, model_file):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model configuration and pretrained model
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("PSUXL/LLMED-combined", trust_remote_code=True, config=config).to(device)

    vecs = tokenizer(vecs, padding="longest", return_tensors="pt")

    embedding = []
    for x, y in zip(vecs["input_ids"], vecs["attention_mask"]):
        hidden = model(x.unsqueeze(0).to(device), attention_mask=y.unsqueeze(0).to(device))[0]
        hidden = hidden.sum(axis=1) / y.sum(axis=-1).to(device)
        embedding.extend(hidden.cpu().data.numpy())
    
    return np.array(embedding)
```

### Experiments
The repository includes code for two experiments, which can be found in the following folders:

correlation_with_edit_distance/
similar_sequence_searching/
Correlation with Edit Distance
This experiment evaluates the correlation between the model's sequence embeddings and the actual edit distance.

Similar Sequence Searching
This experiment demonstrates the ability of the embeddings to retrieve similar sequences from a dataset efficiently.

### Citation
If you use this code or the models in your research, please cite appropriately.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

