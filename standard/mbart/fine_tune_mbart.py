import torch
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq

torch.autograd.set_detect_anomaly(True)

dataset = load_dataset("rahular/itihasa")

train_subset = dataset["train"].shuffle(seed=42).select(range(10000))  
validation_data = dataset["validation"]  
test_data = dataset["test"]  

print(f"Training data size: {len(train_subset)} rows")
print(f"Validation data size: {len(validation_data)} rows")
print(f"Test data size: {len(test_data)} rows")

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX"  
forced_bos_token_id = tokenizer.lang_code_to_id["hi_IN"]  

max_length = 128

def tokenize_function(examples):
    sources = [ex["en"] for ex in examples["translation"]]
    targets = [ex["sn"] for ex in examples["translation"]]
    
    model_inputs = tokenizer(sources, truncation=True, padding="max_length", max_length=max_length)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=max_length).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = train_subset.map(tokenize_function, batched=True)
tokenized_validation = validation_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

training_args = TrainingArguments(
    output_dir="/shared/3/projects/national-culture/cache/independent/cache/mbart",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    run_name="mbart_sanskrit_translation_low_resource"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    padding="max_length", 
    max_length=max_length, 
    label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("/shared/3/projects/national-culture/cache/independent/cache/mbart")
print("Fine-tuning completed and final model saved in '/shared/3/projects/national-culture/cache/independent/cache/mbart'.")

if trainer.state.best_model_checkpoint:
    print(f"Best model checkpoint saved at: {trainer.state.best_model_checkpoint}")