import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)

dataset = load_dataset("rahular/itihasa")
train_subset = dataset["train"].shuffle(seed=42).select(range(10000))
validation_data = dataset["validation"]
test_data = dataset["test"]

print(f"Training data size: {len(train_subset)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

src_lang, tgt_lang, max_length = "eng_Latn", "hin_Deva", 128

def tokenize_function(examples):
    sources = [ex["en"] for ex in examples["translation"]]
    targets = [ex["sn"] for ex in examples["translation"]]
    
    model_inputs = tokenizer(sources, truncation=True, padding="max_length", max_length=max_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, padding="max_length", max_length=max_length).input_ids
    
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in lbl] for lbl in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = train_subset.map(tokenize_function, batched=True)
tokenized_validation = validation_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

training_args = TrainingArguments(
    output_dir="/shared/3/projects/national-culture/cache/independent/cache/indictrans",
    eval_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    fp16=False,  
    push_to_hub=False,
    run_name="indictrans_sanskrit_translation_low_resource"
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

sample_batch = next(iter(trainer.get_train_dataloader()))
sample_batch = {k: v.to(training_args.device) for k, v in sample_batch.items()}

with torch.no_grad():
    outputs = model(**sample_batch)
    print("Forward pass successful, no shape or index errors.")

trainer.train()
trainer.save_model("/shared/3/projects/national-culture/cache/independent/cache/indictrans")
print("Fine-tuning completed and model saved.")