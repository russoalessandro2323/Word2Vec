import json
import os

import numpy as np


def most_similar_global(word, word2idx, idx2word, W_in, top_k=5, verbose=True):
    """Trova le top_k parole più simili a `word` per cosine similarity
    sugli embedding W_in. Ritorna la lista di (parola, similarita).
    """
    word_clean = word.lower()
    if word_clean not in word2idx:
        if verbose:
            print(f"\nParola '{word}' non trovata nel vocabolario.")
        return []

    idx = word2idx[word_clean]
    vec = W_in[idx]

    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-9
    W_norm = W_in / norms
    vec_norm = vec / (np.linalg.norm(vec) + 1e-9)

    sims = W_norm @ vec_norm
    top_indices = np.argsort(sims)[::-1]

    results = []
    if verbose:
        print(f"\nParole più simili a '{word_clean}':")
    for i in top_indices:
        w = idx2word[i]
        if w != word_clean and w not in ["<NWL>", "<UNK>"]:
            results.append((w, float(sims[i])))
            if verbose:
                print(f"  -> {w:<18} (Similarità: {sims[i]:.4f})")
            if len(results) >= top_k:
                break

    return results


def evaluate_topk_accuracy(test_targets, test_contexts, W_in, idx2word,
                            top_k=10, n_samples=200, n_examples_to_show=8,
                            seed=None):
    """Valuta il top-k accuracy sul test set e mostra alcuni esempi
    rappresentativi (hit e miss) per ispezione qualitativa.
    """
    rng = np.random.default_rng(seed)

    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-9
    W_norm = W_in / norms

    n_samples = min(n_samples, len(test_targets))
    sample_idx = rng.choice(len(test_targets), n_samples, replace=False)

    hits = 0
    examples = []

    for i in sample_idx:
        target_idx = test_targets[i]
        context_idx = test_contexts[i]

        vec = W_norm[target_idx]
        sims = W_norm @ vec
        top_indices = np.argsort(sims)[::-1]
        top_indices = top_indices[top_indices != target_idx][:top_k]

        is_hit = context_idx in top_indices
        if is_hit:
            hits += 1

        examples.append({
            "target": idx2word[target_idx],
            "context_reale": idx2word[context_idx],
            "hit": is_hit,
            "top_previsti": [idx2word[j] for j in top_indices[:5]],
        })

    accuracy = hits / n_samples

    print(f"Top-{top_k} accuracy sul test set: {accuracy:.2%} (su {n_samples} campioni)\n")

    hit_examples = [e for e in examples if e["hit"]][:n_examples_to_show // 2]
    miss_examples = [e for e in examples if not e["hit"]][:n_examples_to_show // 2]

    print(f"--- Esempi di HIT (contesto reale tra i top-{top_k}) ---")
    for e in hit_examples:
        print(f"  target: {e['target']:<15} contesto reale: {e['context_reale']:<15} "
              f"top-5 previsti: {e['top_previsti']}")

    print(f"\n--- Esempi di MISS (contesto reale fuori dai top-{top_k}) ---")
    for e in miss_examples:
        print(f"  target: {e['target']:<15} contesto reale: {e['context_reale']:<15} "
              f"top-5 previsti: {e['top_previsti']}")

    return accuracy


def save_embeddings(path, W_in, word2idx, idx2word, word_freq):
    """Salva gli embedding allenati e il vocabolario su disco, in modo da
    poterli riusare (es. come input per una futura RNN) senza dover
    ripetere il training.
    """
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "W_in.npy"), W_in)

    with open(os.path.join(path, "word2idx.json"), "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)

    # idx2word ha chiavi intere: JSON le converte in stringhe, va gestito al caricamento
    with open(os.path.join(path, "idx2word.json"), "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False)

    with open(os.path.join(path, "word_freq.json"), "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in word_freq.items()}, f, ensure_ascii=False)

    print(f"Embedding e vocabolario salvati in: {path}")


def load_embeddings(path):
    """Carica gli embedding e il vocabolario salvati con save_embeddings."""
    W_in = np.load(os.path.join(path, "W_in.npy"))

    with open(os.path.join(path, "word2idx.json"), "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open(os.path.join(path, "idx2word.json"), "r", encoding="utf-8") as f:
        idx2word_raw = json.load(f)
        idx2word = {int(k): v for k, v in idx2word_raw.items()}  # chiavi tornano int

    with open(os.path.join(path, "word_freq.json"), "r", encoding="utf-8") as f:
        word_freq = json.load(f)

    return W_in, word2idx, idx2word, word_freq