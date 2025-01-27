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

Here is an example code snippet to generate embeddings using the PSUXL/LLMED-MAE model:
```
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

# Load DNABERT2 tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")

# Load model
model = AutoModel.from_pretrained("PSUXL/LLMED-MAE", trust_remote_code=True, config=config)

dna = "AGAGCGACGACGTGTAGCAGCTGTACGACTGAGC"

# Get sequence embedding with mean pooling
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
embedding_mean = torch.mean(hidden_states[0], dim=0)
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

