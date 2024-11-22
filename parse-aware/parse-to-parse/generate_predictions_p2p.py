import os
import torch
from datasets import load_dataset
from transformers import MBart50TokenizerFast
from torch.utils.data import DataLoader

from my_multimodal_mbart_model import MultiModalMBart 

def generate_predictions_batch(model, tokenizer, texts, dependency_matrices, device):

    inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=128, truncation=True)
    inputs = inputs.to(device)
    dependency_matrices = dependency_matrices.to(device)

    outputs_text, outputs_tree = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        dependency_matrix=dependency_matrices,
        num_beams=5,
        max_length=256
    )

    decoded_texts = tokenizer.batch_decode(outputs_text, skip_special_tokens=True)
    decoded_trees = outputs_tree.cpu().detach().numpy() 
    return decoded_texts, decoded_trees


def prepare_dependency_matrix(parse_tree, max_length=128):
    dependency_matrix = torch.zeros(max_length, max_length)
    for relation in parse_tree:
        head, dep = relation["head"], relation["dep"]
        if head < max_length and dep < max_length:
            dependency_matrix[head, dep] = 1
    return dependency_matrix


def main():
    model_dir = "./final_multimodal_mbart_model" 
    base_tokenizer_dir = "facebook/mbart-large-50-many-to-many-mmt"

    print("Loading tokenizer from Hugging Face model hub...")
    tokenizer = MBart50TokenizerFast.from_pretrained(base_tokenizer_dir)
    print("Tokenizer loaded successfully.")

    print(f"Loading model from checkpoint directory: {model_dir}")
    model = MultiModalMBart(mbart_model_name=base_tokenizer_dir)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
    model.eval()

    tokenizer.src_lang = "en_XX"  
    print("Loading test dataset from Hugging Face...")
    dataset = load_dataset("rahular/itihasa", split="test")
    print(f"Test dataset loaded. Number of examples: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    batch_size = 64

    predictions_text = []
    predictions_trees = []
    references_text = []
    dependency_matrices = []
    print("Starting prediction generation...")

    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        en_texts = [item['translation']['en'] for item in batch]
        parse_trees = [item.get("en_parse_tree", []) for item in batch] 

        batch_dependency_matrices = torch.stack([
            prepare_dependency_matrix(parse_tree) for parse_tree in parse_trees
        ])

        batch_predictions_text, batch_predictions_tree = generate_predictions_batch(
            model, tokenizer, en_texts, batch_dependency_matrices, device
        )

        predictions_text.extend(batch_predictions_text)
        predictions_trees.extend(batch_predictions_tree)
        references_text.extend(en_texts)

        if (i + batch_size) % 100 < batch_size:
            print(f"Generated predictions for {i + batch_size} examples")

    predictions_text_path = os.path.join(model_dir, 'predicted_texts.txt')
    predictions_tree_path = os.path.join(model_dir, 'predicted_trees.npy') 
    references_path = os.path.join(model_dir, 'references.txt')

    print("Saving predicted texts to file...")
    with open(predictions_text_path, 'w', encoding='utf-8') as pred_file:
        for prediction in predictions_text:
            pred_file.write(prediction + '\n')

    print("Saving predicted dependency trees to file...")
    torch.save(predictions_trees, predictions_tree_path)

    print("Saving references to file...")
    with open(references_path, 'w', encoding='utf-8') as ref_file:
        for ref in references_text:
            ref_file.write(ref + '\n')

    print(f"All predictions saved:\n"
          f"  - Predicted texts: '{predictions_text_path}'\n"
          f"  - Predicted dependency trees: '{predictions_tree_path}'\n"
          f"  - References: '{references_path}'")


if __name__ == "__main__":
    main()