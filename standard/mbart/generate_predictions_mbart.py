import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_dataset

def generate_translation_mbart_batch(model, tokenizer, texts, device):

    inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=128, truncation=True)
    inputs = inputs.to(device)

    translated_tokens = model.generate(**inputs, num_beams=5, max_length=128)
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translations

def main():
    model_dir = "/shared/3/projects/national-culture/cache/independent/cache/mbart/checkpoint-3750"
    base_tokenizer_dir = "facebook/mbart-large-50-many-to-many-mmt"  

    print("Loading tokenizer from Hugging Face model hub...")
    tokenizer = MBart50TokenizerFast.from_pretrained(base_tokenizer_dir)
    print("Tokenizer loaded successfully.")

    print(f"Loading model from checkpoint directory: {model_dir}")
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    tokenizer.src_lang = "hi_IN"  
    tokenizer.tgt_lang = "en_XX"

    print("Loading test dataset from Hugging Face...")
    dataset = load_dataset("rahular/itihasa", split="test")
    print(f"Test dataset loaded. Number of examples: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    batch_size = 64

    predictions = []
    references = []
    print("Starting prediction generation...")
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        sn_texts = [item['translation']['sn'] for item in batch]
        en_references = [item['translation']['en'] for item in batch]
        
        en_predictions = generate_translation_mbart_batch(model, tokenizer, sn_texts, device)
        
        predictions.extend(en_predictions)
        references.extend(en_references)

        if (i + batch_size) % 100 < batch_size:
            print(f"Generated predictions for {i + batch_size} examples")

    predictions_path = os.path.join(model_dir, 'mbart_predictions.txt')
    references_path = os.path.join(model_dir, 'mbart_references.txt')

    print("Saving predictions to file...")
    with open(predictions_path, 'w', encoding='utf-8') as pred_file:
        for pred in predictions:
            pred_file.write(pred + '\n')

    print("Saving references to file...")
    with open(references_path, 'w', encoding='utf-8') as ref_file:
        for ref in references:
            ref_file.write(ref + '\n')

    print(f"mBART predictions have been saved to '{predictions_path}' and references to '{references_path}'.")

if __name__ == "__main__":
    main()