import torch, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

from datasets import load_dataset, load_from_disk


###
### Part 1. Tokenization.
###

# Byte-pair encoding (BPE): the main idea
# ---------------------------------------
# A token is one piece of text (not necessarily a word/subword/"token").
# In this tokenizer, every token is stored as one
# or more bytes. A pair means two tokens that appear next to each other, and a
# merge means joining such a pair to make one longer token.
#
# Training BPE:
#   1. Split every line of the training text into individual UTF-8 bytes.
#   2. Start the vocabulary with the four special tokens and all 256 possible
#      byte values. The vocabulary maps each token to an integer ID.
#   3. Count how often each pair of neighboring tokens occurs.
#   4. Join the most common pair everywhere it occurs. Add the joined token to
#      the vocabulary and save the pair as a merge rule.
#   5. Repeat steps 3 and 4 until the requested vocabulary size is reached 
#      (NB: make sure it is provided).

#
# Encoding new text:
#   1. Remove whitespace from the beginning and end.
#   2. Split the text into individual UTF-8 bytes.
#   3. Apply the saved merge rules in the same order in which they were learned.
#   4. Replace the resulting tokens with their vocabulary IDs.
#   5. Add BOS/EOS IDs, and optionally truncate or pad the sequence.
#
# This simple version treats a whole non-empty line as one sequence. Therefore,
# it may learn tokens that contain spaces or parts of more than one word.

def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.

    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    text_encoded = text.encode('utf-8')
    bytes_list = []

    for byte in text_encoded:
        bytes_list.append(bytes([byte]))

    return bytes_list

def count_pairs(sequences):
    pair_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
    return pair_counts

def merge_tokens(sequences, merge):
    src = merge
    dst = merge[0] + merge[1]

    for seq in sequences:
        i = 0
        while i < len(seq) - 1:
            if (seq[i], seq[i + 1]) == src:
                seq[i:i + 2] = [dst]
                i += 1
            else:
                i += 1

    return sequences

def build_tokenizer(train_file, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """
    

    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).

    with open(train_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    bytes_sequences = [split_into_bytes(line.strip()) for line in texts]

    special_tokens = [pad_token, unk_token, bos_token, eos_token]

    vocabulary = {token: idx for idx, token in enumerate(special_tokens)}


    for byte_idx in range(256):
        vocabulary[bytes([byte_idx])] = len(vocabulary)

    merges = []

    if max_voc_size is not None and max_voc_size < len(vocabulary): # ör detta nödvändigt?
        raise ValueError("Vocabulary size must be at least 260 (4 special tokens + 256 bytes)")

    while max_voc_size is not None and len(vocabulary) < max_voc_size:
        time_start = time.time()

        pair_counts = count_pairs(bytes_sequences)

        if not pair_counts:
            break

        merge = max(pair_counts, key=pair_counts.get)
        merges.append(merge)

        bytes_sequences = merge_tokens(bytes_sequences, merge)

        new_token = merge[0] + merge[1]
        vocabulary[new_token] = len(vocabulary)

        print(f"Merge {len(merges)}: {merge} (count: {pair_counts[merge]}), Time taken: {time.time() - time_start:.2f} seconds")

    return vocabulary, merges

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, vocabulary, merges, model_max_length, pad_token='<PAD>',
                 unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        # TODO: store all values you need in order to implement __call__ below.
        self.vocabulary = vocabulary
        self.inverse_vocabulary = {v: k for k, v in vocabulary.items()}
        self.merges = merges

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.pad_token_id = vocabulary[pad_token]  # Compulsory attribute.
        self.unk_token_id = vocabulary[unk_token]
        self.bos_token_id = vocabulary[bos_token]
        self.eos_token_id = vocabulary[eos_token]

        self.model_max_length = model_max_length  # Needed for truncation.

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        byte_sequences = [split_into_bytes(text.strip()) for text in texts]
        for merge in self.merges:
            byte_sequences = merge_tokens(byte_sequences, merge)
        encoded_sequences = []
        for seq in byte_sequences:
            encoded_seq = []
            for token in seq:
                if token in self.vocabulary:
                    encoded_seq.append(self.vocabulary[token])
                else:
                    encoded_seq.append(self.vocabulary[self.unk_token])
            encoded_sequences.append(encoded_seq)
        

        if truncation and self.model_max_length > 2:
            encoded_sequences = [seq[:self.model_max_length-2] for seq in encoded_sequences] ### Fråga om BOS EOS ska räknas med i max L, #måste vara större än 2 för att kunna lägga till BOS och EOS

        encoded_sequences = [[self.vocabulary[self.bos_token]] + seq + [self.vocabulary[self.eos_token]] for seq in encoded_sequences] 

        if padding:
            max_length = max(len(seq) for seq in encoded_sequences)
            encoded_sequences = [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in encoded_sequences]
            attention_mask = [[1 if token_id != self.pad_token_id else 0 for token_id in seq] for seq in encoded_sequences]
        else:
            attention_mask = [[1] * len(seq) for seq in encoded_sequences]

        if return_tensors == "pt":
            encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

        return BatchEncoding({'input_ids': encoded_sequences, 'attention_mask': attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def decode(self, ids):
        """Return a string decoded from given input ids"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special_token_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for id in ids:
            if id in special_token_ids:
                continue
            if id in self.inverse_vocabulary:
                token = self.inverse_vocabulary[id]
            else:
                token = self.inverse_vocabulary[self.unk_token_id]
            tokens.append(token)
        text_bytes = b''.join(tokens)
        decoded_text = text_bytes.decode('utf-8')
        #kan ett okänt token_id finnas med ids? vad händer då?
        return decoded_text


    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
# Part 2. Loading the dataset.
texts = ["hue is good", "test this also"]

model_max_length = 512

train_filename = r"train.txt"
run_tokenizer = False
if run_tokenizer:
    tokenizer = build_tokenizer(train_filename, max_voc_size=10000, model_max_length=model_max_length)
    tokenizer.save("tokenizer.pkl")
else:
    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")

token_ids = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
print(token_ids)
decoded_texts = [tokenizer.decode(ids) for ids in token_ids['input_ids']]
print(decoded_texts)

train_file = r'train.txt'
val_file = r'val.txt'
dataset = load_dataset('text', data_files={'train': train_file, 'validation': val_file})
dataset = dataset.filter(lambda x: (x['text'].strip()) != '') # remove empty lines

print(f"Length of training set: {len(dataset['train'])}, Length of validation set: {len(dataset['validation'])}")

tokenized_dataset_boolvariable = True
if tokenized_dataset_boolvariable:
    tokenized_dataset = dataset.map(
        lambda batch: dict(tokenizer(
            batch['text'],
            truncation=True,
            padding=True
        )),
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=['text']
    )
    tokenized_dataset.save_to_disk('tokenized_dataset_padded')
else:
    tokenized_dataset = load_from_disk('tokenized_dataset')
