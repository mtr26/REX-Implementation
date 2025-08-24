import argparse
import os
import torch
from transformers import AutoTokenizer
from model import Transformer
import json
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Sequence, Value, Dataset
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

MODEL = 'mistralai/Mistral-7B-v0.3'

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    PAD_ID = tokenizer.pad_token_id

BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
special_tokens = [f"<extra_id_{i}>" for i in range(100)]
num_new = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

sentinel_start = tokenizer.convert_tokens_to_ids("<extra_id_0>")

class Config:
    def __init__(
            self,
            dim: int,
            encoder_layers: int,
            decoder_layers: int,
            num_heads: int,
            max_length: int,
            latent_dim: int,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,
            eval_steps: int
            ):
        self.dim = dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_steps = eval_steps


def span_corrupt(token_ids, noise_density=0.15, mean_span_len=3, extra_id_base=sentinel_start):
    n_tokens = len(token_ids)
    n_mask = max(1, int(n_tokens * noise_density))

    # choose random start points for spans
    span_starts = np.random.choice(range(n_tokens), size=n_mask, replace=False)
    span_starts.sort()

    enc, dec = [], []
    cursor, sentinel_id = 0, extra_id_base
    for start in span_starts:
        if start < cursor:
            continue
        span_len = np.random.poisson(mean_span_len)
        span_len = max(1, min(span_len, n_tokens - start))

        # keep before
        enc.extend(token_ids[cursor:start])
        # sentinel in encoder
        enc.append(sentinel_id)

        # decoder: sentinel + removed tokens
        dec.append(sentinel_id)
        dec.extend(token_ids[start:start+span_len])

        cursor = start + span_len
        sentinel_id -= 1

    # tail
    enc.extend(token_ids[cursor:])
    dec.append(sentinel_id)

    return enc, dec


def pad_to_seq_len(x, seq_len, pad_id):
    if len(x) < seq_len:
        return x + [pad_id] * (seq_len - len(x))
    else:
        return x[:seq_len]

def make_attention_mask(seq, pad_id=PAD_ID):
    return [1 if t != pad_id else 0 for t in seq]

def chunk_and_pack_for_distill_span(examples, seq_len=512):
    all_ids = sum(
        tokenizer(examples["text"], add_special_tokens=False)["input_ids"], []
    )

    chunks = [
        all_ids[i : i + seq_len]
        for i in range(0, len(all_ids) - seq_len, seq_len)
    ]

    student_encoder_input_ids, student_decoder_input_ids, student_labels = [], [], []
    student_encoder_attention_mask, student_decoder_attention_mask = [], []

    for c in chunks:
        enc, dec = span_corrupt(c)

        enc_p  = pad_to_seq_len(enc, seq_len, PAD_ID)
        dec_in = pad_to_seq_len([BOS_ID] + dec[:-1], seq_len, PAD_ID)
        dec_lab= pad_to_seq_len(dec + [EOS_ID], seq_len, -100)

        student_encoder_input_ids.append(enc_p)
        student_decoder_input_ids.append(dec_in)
        student_labels.append(dec_lab)

        student_encoder_attention_mask.append(make_attention_mask(enc_p))
        student_decoder_attention_mask.append(make_attention_mask(dec_in))

    return {
        "input_ids": student_encoder_input_ids,
        "decoder_input_ids": student_decoder_input_ids,
        "labels":            student_labels,
        "attention_mask": student_encoder_attention_mask,
        "decoder_attention_mask": student_decoder_attention_mask,
    }

def load_data(path, tokenizer, seq_len=512, ratio=0.01):
    dataset = load_dataset("json", data_files=path, split="train")

    total_len = len(dataset)
    val_size = int(total_len * ratio)

    # Validation slice (first 1%)
    val_dataset = dataset.select(range(val_size))

    # Training slice (remaining 99.9%)
    train_dataset = dataset.select(range(val_size, total_len))
    
    
    train_dataset = train_dataset.map(
        lambda x: chunk_and_pack_for_distill_span(x, seq_len=seq_len),
        batched=True,
        remove_columns=["text"],
    ).with_format("torch")

    val_dataset = val_dataset.map(
        lambda x: chunk_and_pack_for_distill_span(x, seq_len=seq_len),
        batched=True,
        remove_columns=["text"],
    ).with_format("torch")

    return train_dataset, val_dataset



def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    os.environ['HF_TOKEN'] = config_data['api_key']
    return Config(
        dim=config_data['dim'],
        encoder_layers=config_data['encoder_layers'],
        decoder_layers=config_data['decoder_layers'],
        num_heads=config_data['num_heads'],
        max_length=config_data['max_length'],
        latent_dim=config_data['latent_dim'],
        batch_size=config_data['batch_size'],
        learning_rate=config_data['learning_rate'],
        weight_decay=config_data['weight_decay'],
        eval_steps=config_data['eval_steps']
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    train_dataset, val_dataset = load_data(args.data_path, tokenizer, seq_len=config.max_length, ratio=0.01)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./rex_output",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        save_strategy="steps",
        report_to=[],
        save_steps=config.eval_steps,
        logging_steps=config.eval_steps,
        eval_steps=config.eval_steps,
        save_total_limit=3,
        bf16=True,                   # use mixed precision if your TPU/GPU supports
        gradient_accumulation_steps=1,
        num_train_epochs=1
    )

    model = Transformer(
        dim=config.dim,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        max_length=config.max_length,
        latent_dim=config.latent_dim
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    

