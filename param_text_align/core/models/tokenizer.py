from transformers import BertTokenizer
import torch
from torch.nn import Linear, ReLU, Sequential

model = Sequential(
  Linear(in_features=784, out_features=50, bias=True),
  ReLU(),
  Linear(in_features=50, out_features=25, bias=True),
  ReLU(),
  Linear(in_features=25, out_features=10, bias=True),
)



def arch_tokenizer(model):
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = torch.empty(1, 0)
    token_type_ids = torch.empty(1, 0)
    attention_mask = torch.empty(1, 0)

    layer_causality_encoding = []

    for layer_id in range(len(model)):
        # use text_tokenizer to encode the layer string but without CLS and SEP tokens
        tokens_layer = text_tokenizer(str(model[layer_id]), return_tensors='pt', padding=True, truncation=True)

        input_ids = torch.cat((input_ids, tokens_layer['input_ids'][:, 1:-1]), dim=1)
        token_type_ids = torch.cat((token_type_ids, tokens_layer['token_type_ids'][:, 1:-1]), dim=1)
        attention_mask = torch.cat((attention_mask, tokens_layer['attention_mask'][:, 1:-1]), dim=1)

        layer_causality_encoding += [layer_id] * tokens_layer['input_ids'][:, 1:-1].shape[1]

    # add final SEP token
    input_ids = torch.cat((input_ids,  torch.Tensor([[102]])), dim=1)
    token_type_ids = torch.cat((token_type_ids, torch.Tensor([[0]])), dim=1)
    attention_mask = torch.cat((attention_mask, torch.Tensor([[1]])), dim=1)
    layer_causality_encoding = torch.tensor(layer_causality_encoding + [layer_causality_encoding[-1]]).reshape(1, -1)

    return {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}, layer_causality_encoding




def custom_tokenizer(model, layer_id, text_task='image classification', text_dataset='MNIST'):
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_layer = "Layer (" + str(layer_id) + "): " + str(model[layer_id])

    tokens_layer = text_tokenizer(text_layer, return_tensors='pt', padding=True, truncation=True)
    tokens_arch, layer_causality_encoding = arch_tokenizer(model)
    tokens_task = text_tokenizer(text_task, return_tensors='pt', padding=True, truncation=True)
    tokens_dataset = text_tokenizer(text_dataset, return_tensors='pt', padding=True, truncation=True)

    input_ids = torch.cat((tokens_layer['input_ids'], tokens_arch['input_ids'][:, 1:], tokens_task['input_ids'][:, 1:], tokens_dataset['input_ids'][:, 1:]), dim=1)
    # token_type_ids = torch.cat((tokens_layer['token_type_ids'], tokens_arch['token_type_ids'][:, 1:], tokens_task['token_type_ids'][:, 1:], tokens_dataset['token_type_ids'][:, 1:]), dim=1)
    # attention_mask = torch.cat((tokens_layer['attention_mask'], tokens_arch['attention_mask'][:, 1:], tokens_task['attention_mask'][:, 1:], tokens_dataset['attention_mask'][:, 1:]), dim=1)

    layer_encoding = torch.tensor([0] * tokens_layer['input_ids'].shape[1]).reshape(1, -1)
    arch_encoding = torch.tensor([1] * tokens_arch['input_ids'].shape[1]).reshape(1, -1)
    task_encoding = torch.tensor([2] * tokens_task['input_ids'].shape[1]).reshape(1, -1)
    dataset_encoding = torch.tensor([3] * tokens_dataset['input_ids'].shape[1]).reshape(1, -1)

    sentence_encoding = torch.cat((layer_encoding, arch_encoding, task_encoding, dataset_encoding), dim=1)

    # return {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}, sentence_encoding, layer_causality_encoding
    return input_ids, sentence_encoding, layer_causality_encoding
