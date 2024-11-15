import argparse
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_from_disk
import torch

def rouge_n(candidate, references, n=1):
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]

    def get_ngrams(tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = []
    for ref_tokens in reference_tokens:
        reference_ngrams.extend(get_ngrams(ref_tokens, n))

    overlapping_ngrams = set(candidate_ngrams) & set(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    precision = overlapping_count / len(set(candidate_ngrams)) if candidate_ngrams else 0
    recall = overlapping_count / len(set(reference_ngrams)) if reference_ngrams else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1_score}

def meteor(candidate, references):
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]

    if not any(reference_tokens): 
        return 0

    matches = sum(1 for ref in reference_tokens for w in candidate_tokens if w in ref)
    precision = matches / len(candidate_tokens) if candidate_tokens else 0
    total_ref_len = sum(len(ref) for ref in reference_tokens)
    recall = matches / total_ref_len if total_ref_len > 0 else 0

    if precision + recall > 0:
        f_mean = (10 * precision * recall) / (9 * precision + recall)
    else:
        f_mean = 0

    penalty = 0.5 * (len(candidate_tokens) / total_ref_len) if total_ref_len > 0 else 0
    meteor_score = f_mean * (1 - penalty)
    return meteor_score

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def ter(candidate, references):
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]

    distances = []
    for ref_tokens in reference_tokens:
        distance = levenshtein_distance(candidate_tokens, ref_tokens)
        distances.append(distance / len(ref_tokens) if ref_tokens else 0)

    return min(distances)

def chrf(candidate, references, n=6, beta=2):
    def get_char_ngrams(text, n):
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams

    total_precisions = []
    total_recalls = []
    for k in range(1, n+1):
        cand_ngrams = get_char_ngrams(candidate.replace(' ', ''), k)
        ref_ngrams = []
        for ref in references:
            ref_ngrams.extend(get_char_ngrams(ref.replace(' ', ''), k))

        cand_ngram_set = set(cand_ngrams)
        ref_ngram_set = set(ref_ngrams)

        overlap = cand_ngram_set & ref_ngram_set
        precision = len(overlap) / len(cand_ngram_set) if cand_ngram_set else 0
        recall = len(overlap) / len(ref_ngram_set) if ref_ngram_set else 0

        total_precisions.append(precision)
        total_recalls.append(recall)

    avg_precision = sum(total_precisions) / n
    avg_recall = sum(total_recalls) / n

    if avg_precision + avg_recall > 0:
        f_score = ((1 + beta**2) * avg_precision * avg_recall) / (beta**2 * avg_precision + avg_recall)
    else:
        f_score = 0

    return f_score

def wer(candidate, reference):
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    d = [[0] * (len(reference_tokens) + 1) for _ in range(len(candidate_tokens) + 1)]
    for i in range(len(candidate_tokens) + 1):
        d[i][0] = i
    for j in range(len(reference_tokens) + 1):
        d[0][j] = j

    for i in range(1, len(candidate_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
    wer_score = d[len(candidate_tokens)][len(reference_tokens)] / len(reference_tokens) if reference_tokens else 0
    return wer_score

def evaluate_model(model_dir, test_dataset_dir):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    print(f"Tokenizer and model loaded from '{model_dir}'.")

    tokenizer.src_lang = "hi_IN"
    tokenizer.tgt_lang = "en_XX"

    datasets = load_from_disk(test_dataset_dir)
    test_dataset = datasets['test']
    print("Test dataset loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_translation(sn_text):
        sn_text = sn_text.strip()
        inputs = tokenizer(sn_text, return_tensors="pt", max_length=128, truncation=True)
        inputs = inputs.to(device)
        translated_tokens = model.generate(**inputs, num_beams=5, max_length=128)
        en_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return en_text

    print("Evaluating the model on the test set...")
    predictions = []
    references = []

    total_rouge1_f1 = 0
    total_rouge2_f1 = 0
    total_meteor = 0
    total_ter = 0
    total_chrf = 0
    total_wer = 0
    total_levenshtein = 0

    for i, item in enumerate(test_dataset):
        sn_text = item['sn']
        en_reference = item['en'].strip()
        en_prediction = generate_translation(sn_text)
        predictions.append(en_prediction.strip())
        references.append(en_reference)

        rouge1 = rouge_n(en_prediction, [en_reference], n=1)
        rouge2 = rouge_n(en_prediction, [en_reference], n=2)
        total_rouge1_f1 += rouge1['f1']
        total_rouge2_f1 += rouge2['f1']

        meteor_score = meteor(en_prediction, [en_reference])
        total_meteor += meteor_score

        ter_score = ter(en_prediction, [en_reference])
        total_ter += ter_score

        chrf_score = chrf(en_prediction, [en_reference])
        total_chrf += chrf_score

        wer_score = wer(en_prediction, en_reference)
        total_wer += wer_score

        distance = levenshtein_distance(en_prediction, en_reference)
        normalized_distance = distance / max(len(en_prediction), len(en_reference)) if max(len(en_prediction), len(en_reference)) > 0 else 0
        total_levenshtein += normalized_distance

    n = len(test_dataset)
    avg_rouge1_f1 = (total_rouge1_f1 / n) * 100
    avg_rouge2_f1 = (total_rouge2_f1 / n) * 100
    avg_meteor = (total_meteor / n) * 100
    avg_ter = (total_ter / n) * 100
    avg_chrf = (total_chrf / n) * 100
    avg_wer = (total_wer / n) * 100
    avg_levenshtein = (total_levenshtein / n) * 100

    print(f"Average ROUGE-1 F1 Score: {avg_rouge1_f1:.2f}")
    print(f"Average ROUGE-2 F1 Score: {avg_rouge2_f1:.2f}")
    print(f"Average METEOR Score: {avg_meteor:.2f}")
    print(f"Average TER Score: {avg_ter:.2f}")
    print(f"Average ChrF Score: {avg_chrf:.2f}")
    print(f"Average WER Score: {avg_wer:.2f}")
    print(f"Average Normalized Levenshtein Distance: {avg_levenshtein:.2f}")

    with open('predictions.txt', 'w', encoding='utf-8') as pred_file:
        for pred in predictions:
            pred_file.write(pred + '\n')

    with open('references.txt', 'w', encoding='utf-8') as ref_file:
        for ref in references:
            ref_file.write(ref + '\n')

    print("Translations have been saved to 'predictions.txt' and 'references.txt'.")

    sn_example = "ॐ तपः स्वाध्यायनिरतं तपस्वी वाग्विदां वरम्।"
    en_translation = generate_translation(sn_example)
    print(f"\nSanskrit Text: {sn_example}")
    print(f"Translated Text: {en_translation}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the translation model")

    parser.add_argument('--model_dir', type=str, required=True, help="Directory where the model is saved")
    parser.add_argument('--test_dataset_dir', type=str, required=True, help="Directory where the test dataset is saved")

    args = parser.parse_args()

    evaluate_model(model_dir=args.model_dir, test_dataset_dir=args.test_dataset_dir)