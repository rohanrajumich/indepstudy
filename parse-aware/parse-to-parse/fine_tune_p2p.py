import os
import sys
import torch
from torch import nn
from datasets import load_dataset
from transformers import MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers.models.mbart.modeling_mbart import MBartModel


# fusion Layer for cross-attention between text and dependency embeddings
class FusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, text_embeddings, dep_embeddings, attention_mask):
        # cross-attend dependency embeddings with text embeddings
        fused_output, _ = self.attention(
            query=text_embeddings,
            key=dep_embeddings,
            value=dep_embeddings,
            key_padding_mask=~attention_mask.bool()  # Mask for valid positions
        )
        # concatenate original text embeddings and attended dependency embeddings, then project
        fused_output = torch.cat([text_embeddings, fused_output], dim=-1)
        return self.fc(fused_output)


class MultiModalMBart(nn.Module):
    def __init__(self, mbart_model_name="facebook/mbart-large-50-many-to-many-mmt"):
        super().__init__()
        self.mbart = MBartModel.from_pretrained(mbart_model_name)
        self.fusion_layer = FusionLayer(self.mbart.config.d_model)

        self.text_decoder = self.mbart.decoder  
        self.tree_decoder = nn.Linear(self.mbart.config.d_model, 128)  

    def forward(self, input_ids, attention_mask, dependency_matrix, decoder_input_ids, decoder_attention_mask):
        text_outputs = self.mbart.encoder(input_ids=input_ids, attention_mask=attention_mask)

        dep_embeddings = nn.Linear(dependency_matrix.size(-1), text_outputs.last_hidden_state.size(-1)).to(input_ids.device)(dependency_matrix)

        # fusing text and dependency embeddings
        fused_encoder_outputs = self.fusion_layer(text_outputs.last_hidden_state, dep_embeddings, attention_mask)

        text_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_encoder_outputs,
            encoder_attention_mask=attention_mask
        )

        tree_outputs = self.tree_decoder(fused_encoder_outputs)

        return text_outputs, tree_outputs


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)

dataset = load_dataset("rahular/itihasa")
train_subset = dataset["train"].shuffle(seed=42).select(range(10000))
validation_data = dataset["validation"]
test_data = dataset["test"]

print(f"Training data size: {len(train_subset)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX" 
tokenizer.tgt_lang = "hi_IN" 
max_length = 128


# preparing dependency tree embeddings
def prepare_dependency_matrix(parse_tree, max_length=128):
    dependency_matrix = torch.zeros(max_length, max_length)
    for relation in parse_tree:
        head, dep = relation["head"], relation["dep"]
        if head < max_length and dep < max_length:
            dependency_matrix[head, dep] = 1
    return dependency_matrix


def tokenize_function(examples):
    sources = [ex["en"] for ex in examples["translation"]]
    targets = [ex["sn"] for ex in examples["translation"]]
    parse_trees = [ex.get("en_parse_tree", []) for ex in examples["translation"]]

    model_inputs = tokenizer(sources, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").input_ids

    dependency_matrices = [prepare_dependency_matrix(tree, max_length=max_length) for tree in parse_trees]

    model_inputs["labels"] = labels.tolist()
    model_inputs["dependency_matrix"] = torch.stack(dependency_matrices)
    return model_inputs

print("Tokenizing train dataset...")
tokenized_train = train_subset.map(tokenize_function, batched=True)
print("Tokenizing validation dataset...")
tokenized_validation = validation_data.map(tokenize_function, batched=True)
print("Tokenizing test dataset...")
tokenized_test = test_data.map(tokenize_function, batched=True)

model = MultiModalMBart()

training_args = Seq2SeqTrainingArguments(
    output_dir="./model_outputs",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
)

def compute_loss(pred_text, pred_tree, target_text, target_tree):
    # loss for text generation
    text_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    text_loss = text_loss_fn(pred_text.view(-1, pred_text.size(-1)), target_text.view(-1))

    # loss for dependency tree prediction
    tree_loss_fn = nn.MSELoss() 
    tree_loss = tree_loss_fn(pred_tree, target_tree)

    total_loss = 0.7 * text_loss + 0.3 * tree_loss
    return total_loss

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
    max_length=max_length,
    label_pad_token_id=-100
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator
)

sample_batch = next(iter(trainer.get_train_dataloader()))
sample_batch = {k: v.to(training_args.device) for k, v in sample_batch.items()}

with torch.no_grad():
    pred_text, pred_tree = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        dependency_matrix=sample_batch["dependency_matrix"],
        decoder_input_ids=sample_batch["labels"],
        decoder_attention_mask=sample_batch["attention_mask"],
    )
    print("Forward pass successful, no shape or index errors.")

trainer.train()
trainer.save_model("./final_multimodal_mbart_model")
print("Training complete and model saved.")