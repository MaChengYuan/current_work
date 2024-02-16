import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForMaskedLM

from transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL = 'bert-base-uncased'
logger = logging.getLogger(__name__)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


 
      
def convert_examples_to_features(examples,examples_labels, label_list, max_seq_length, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    # ----
    dupe_factor = 5
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(seed)

    for (ex_index, example) in enumerate(zip(examples,examples_labels)):
        modified_example = example[0] + " " + example[1]
        tokens_a = tokenizer.tokenize(modified_example)

        
        segment_id = label_map[example[1]]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(segment_id)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(segment_id)
        tokens.append("[SEP]")
        segment_ids.append(segment_id)
        masked_lm_labels = [-100] * max_seq_length

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        len_cand = len(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms_pos = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms_pos) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]

            masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
            output_tokens[index] = masked_token
            masked_lms_pos.append(index)

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            init_ids.append(0)
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)  # ?segment_id

        assert len(init_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
            InputFeatures(init_ids=init_ids,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          masked_lm_labels=masked_lm_labels))
    return features



def prepare_data(features):
    all_init_ids = torch.tensor([f.init_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features],
                                        dtype=torch.long)
    tensor_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_segment_ids,
                               all_masked_lm_labels)
    return tensor_data



def rev_wordpiece(str):
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]' or str[i] == "[SEP]" or str[i] == '[CLS]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[1:-1])
    
def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        _, input_ids, input_mask, segment_ids, masked_ids = batch
        inputs = {'input_ids': batch[1],
                  'attention_mask': batch[2],
                  'token_type_ids': batch[3],
                  'labels': batch[4]}

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss



def augment_train_data(model, tokenizer, train_data, label_list, train_batch_size , output_dir,file_name , sample_ratio , temp , sample_num):
    # load the best model
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=train_batch_size)
    best_model_path = os.path.join(output_dir, 'best_cbert.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise ValueError("Unable to find the saved model at {}".format(best_model_path))

    save_train_path = os.path.join(output_dir,file_name)
    save_train_file = open(save_train_path, 'w')

    MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    tsv_writer = csv.writer(save_train_file, delimiter='\t')
    tsv_writer.writerow(['label','sentence'])

    print('generating')
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        init_ids, _, input_mask, segment_ids, _ = batch
        input_lens = [sum(mask).item() for mask in input_mask]

        x = [np.random.randint(0, l, max(l // sample_ratio, 1)) for l in input_lens]
        masked_idx = []
        for row in x:
            masked_idx.extend(row)
        masked_idx = np.squeeze(masked_idx)
        for ids, idx in zip(init_ids, masked_idx):
            ids[idx] = MASK_id

        inputs = {'input_ids': init_ids,
                  'attention_mask': input_mask,
                  'token_type_ids': segment_ids}
        outputs = model(**inputs)
        predictions = outputs[0]  # model(init_ids, segment_ids, input_mask)
        predictions = F.softmax(predictions / temp, dim=2)

        for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
            preds = torch.multinomial(preds, sample_num, replacement=True)[idx]
            if len(preds.size()) == 2:
                preds = torch.transpose(preds, 0, 1)
            for pred in preds:
                ids[idx] = pred
                new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
                new_str = rev_wordpiece(new_str)
                tsv_writer.writerow([label_list[seg[0].item()], new_str])



def train_cbert_and_augment(model, tokenizer,train_df,val_df,output='cbert',file_name='augment.tsv',seed = 1234,max_seq_length = 64,sample_num=6,num_train_epochs=8):
    seed = seed
    num_train_epochs = num_train_epochs
    train_batch_size = 8 
    learning_rate = 4e-5
    max_seq_length = max_seq_length
    output_dir = os.path.join(output, file_name)
    sample_ratio , temp , sample_num = 7 , 1 , sample_num


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(output, exist_ok=True)


    # load train and dev data

   
    model.to(device)

    train_df['label'] = train_df['label'].apply(lambda x:str(x))
    train_examples = train_df['sentence']
    
    train_label = train_df['label']
    
    label_list = set()
    for i in train_label:
        label_list.add(i)
    label_list = sorted(label_list)

    if len(label_list) > 2:
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(len(label_list), 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        
    train_features = convert_examples_to_features(train_examples,train_label, label_list,
                                                  max_seq_length,
                                                  tokenizer, seed)
    train_data = prepare_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=train_batch_size)

    print('dev')

    val_df['label'] = val_df['label'].apply(lambda x:str(x))
    val_examples = val_df['sentence']
    
    val_label = val_df['label']
    
    #dev data
    dev_features = convert_examples_to_features(val_examples,val_label, label_list,
                                                  max_seq_length,
                                                  tokenizer, seed)
    dev_data = prepare_data(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=train_batch_size)



    
    num_train_steps = int(len(train_features) / train_batch_size * num_train_epochs)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    # Prepare optimizer
    t_total = num_train_steps
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    best_dev_loss = float('inf')

    print('cbert training')
    for epoch in trange(int(num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[1],
                      'attention_mask': batch[2],
                      'token_type_ids': batch[3],
                      'labels': batch[4]}

            outputs = model(**inputs)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
            #avg_loss = 0.

        dev_loss = compute_dev_loss(model, dev_dataloader)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_model_path = os.path.join(output, 'best_cbert.pt')
            torch.save(model.state_dict(), save_model_path)

    # augment data using the best model

    
    augment_train_data(model, tokenizer, train_data, label_list, train_batch_size , output,file_name , sample_ratio , temp , sample_num)

