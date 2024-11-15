import os
import torch
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)

try:
    import spacy
    spacy_version = spacy.__version__
    print(f"spaCy version: {spacy_version}")
except ImportError:
    raise ImportError("spaCy is not installed. Please install it using 'pip install spacy' and 'python -m spacy download en_core_web_sm'.")

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy 'en_core_web_sm' model loaded successfully.")
except OSError:
    raise OSError("The spaCy model 'en_core_web_sm' is not installed. Please install it using 'python -m spacy download en_core_web_sm'.")

print("Loading the dataset...")
dataset = load_dataset("rahular/itihasa")
print("Dataset loaded successfully.")

train_subset = dataset["train"].shuffle(seed=42).select(range(10000))
validation_data = dataset["validation"]

print(f"Training data size: {len(train_subset)} rows")
print(f"Validation data size: {len(validation_data)} rows")

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX"
max_length = 128

def generate_parse_tree(text):
    doc = nlp(text)
    parse_tree = []
    for token in doc:
        parse_tree.append(f"{token.text}/{token.dep_}/{token.head.text}")
    return " ".join(parse_tree)

def add_parse_tree_column(example):
    english_text = example["translation"]["en"]
    parse_tree_text = generate_parse_tree(english_text)
    example["parse_tree"] = parse_tree_text
    return example

print("Adding parse tree column to dataset...")
dataset = dataset.map(add_parse_tree_column, batched=False)

print("\nExample data with parse tree:")
for example in dataset["train"].select(range(2)):
    print(example)

def tokenize_translation_task(example):
    english_text = example["translation"]["en"]

    # Translation task: English to Sanskrit
    translation_target = tokenizer(
        example["translation"]["sn"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    ).input_ids

    # Input (English sentence)
    input_encoding = tokenizer(
        english_text, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    
    # Package data for each example
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": translation_target
    }

print("Tokenizing dataset for translation task...")
tokenized_train = train_subset.map(tokenize_translation_task, batched=False)
tokenized_validation = validation_data.map(tokenize_translation_task, batched=False)
print("Tokenization completed.")

print("Loading the model...")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
print("Model loaded successfully.")

training_args = Seq2SeqTrainingArguments(
    output_dir="/shared/3/projects/national-culture/cache/independent/cache/input-to-parse",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    run_name="mbart_translation_task"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    padding="max_length", 
    max_length=max_length, 
    label_pad_token_id=-100
)

print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

trainer.save_model("/shared/3/projects/national-culture/cache/independent/cache/mbart_translation")
print("Fine-tuning completed and final model saved.")

if trainer.state.best_model_checkpoint:
    print(f"Best model checkpoint saved at: {trainer.state.best_model_checkpoint}")