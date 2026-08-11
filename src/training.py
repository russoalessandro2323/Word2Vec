import os
import time

import numpy as np

from .layers import DenseLayer, EmbeddingLayer, SoftMaxCrossEntropy
from .preprocessing import build_vocab, encode, generate_training_pairs


def load_corpus(file_corpus="corpus_wikipedia.txt", n_articles=300,
                 dataset_name="wikimedia/wikipedia", dataset_config="20231101.it"):
    """Carica il corpus da file locale se presente, altrimenti lo scarica
    da HuggingFace e lo salva per i run successivi.
    """
    if os.path.exists(file_corpus):
        print("Caricamento del corpus dal file locale...")
        with open(file_corpus, "r", encoding="utf-8") as f:
            return f.read()

    print("Download del dataset da HuggingFace in corso...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=f"train[:{n_articles}]")
    corpus = "\n".join(dataset["text"])

    with open(file_corpus, "w", encoding="utf-8") as f:
        f.write(corpus)
    print("Corpus salvato localmente!")

    return corpus


def to_one_hot(indices, vocab_size):
    oh = np.zeros((len(indices), vocab_size))
    oh[np.arange(len(indices)), indices] = 1.0
    return oh


def prepare_training_data(corpus, window_size=3, test_split=0.1, seed=None):
    """Costruisce vocabolario, encoding e coppie target-contesto, poi le
    divide in training/test con uno shuffle preventivo (per evitare che
    il test set contenga solo le coppie degli ultimi articoli del corpus).
    """
    word2idx, idx2word, word_freq = build_vocab(corpus)
    corpus_indices = encode(corpus, word2idx)
    word_probs = { key : value / len(corpus_indices) for key, value in word_freq.items() }
    pairs = generate_training_pairs(corpus_indices, idx2word, word_probs, 3)

    pairs_arr = np.array(pairs)
    targets = pairs_arr[:, 0]
    contexts = pairs_arr[:, 1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(targets))
    shuffled_targets = targets[perm]
    shuffled_contexts = contexts[perm]

    idx_split = int(np.ceil(len(targets) * (1 - test_split)))
    data = {
        "train_targets": shuffled_targets[:idx_split],
        "test_targets": shuffled_targets[idx_split:],
        "train_contexts": shuffled_contexts[:idx_split],
        "test_contexts": shuffled_contexts[idx_split:],
    }

    return data, word2idx, idx2word, word_freq


def train_word2vec(train_targets, train_contexts, vocab_size, embedding_dim=50,
                    epochs=5, batch_size=1024, learning_rate=0.3, seed=None,
                    verbose=True):
    """Allena EmbeddingLayer + DenseLayer + SoftMaxCrossEntropy con SGD
    mini-batch. Ritorna i layer allenati (l'embedding importante è
    embed_layer.w).
    """
    rng = np.random.default_rng(seed)

    embed_layer = EmbeddingLayer(vocab_size, embedding_dim)
    dense_layer = DenseLayer(embedding_dim, vocab_size)
    loss_layer = SoftMaxCrossEntropy()

    num_samples = len(train_targets)
    num_batches_per_epoch = int(np.ceil(num_samples / batch_size))

    for epoch in range(epochs):
        perm = rng.permutation(num_samples)
        shuffled_targets = train_targets[perm]
        shuffled_contexts = train_contexts[perm]

        epoch_loss = 0.0
        epoch_t0 = time.time()

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_targets = shuffled_targets[start:end]
            batch_contexts = shuffled_contexts[start:end]
            y_true = to_one_hot(batch_contexts, vocab_size)

            h = embed_layer.forward(batch_targets)
            z = dense_layer.forward(h)
            loss = loss_layer.forward(z, y_true)

            grad_loss = loss_layer.backward()
            grad_dense = dense_layer.backward(grad_loss)
            embed_layer.backward(grad_dense)

            dense_layer.w -= learning_rate * dense_layer.dw
            dense_layer.b -= learning_rate * dense_layer.db
            embed_layer.w -= learning_rate * embed_layer.dw

            epoch_loss += loss

        avg_loss = epoch_loss / num_batches_per_epoch
        if verbose:
            elapsed = time.time() - epoch_t0
            print(f"Epoch {epoch + 1}/{epochs} - Loss media: {avg_loss:.4f} - Tempo: {elapsed:.1f}s")

    return embed_layer, dense_layer, loss_layer


def main():
    corpus = load_corpus()
    print(f"Parole totali (grezze): {len(corpus.split())}")

    data, word2idx, idx2word, word_freq = prepare_training_data(corpus, window_size=3)
    vocab_size = len(word2idx)
    print(f"Vocabolario: {vocab_size} parole. Coppie di training: {len(data['train_targets'])}")

    embed_layer, dense_layer, loss_layer = train_word2vec(
        data["train_targets"], data["train_contexts"], vocab_size,
        embedding_dim=50, epochs=5, batch_size=1024, learning_rate=0.3,
    )

    from .evaluation import evaluate_topk_accuracy, save_embeddings

    evaluate_topk_accuracy(data["test_targets"], data["test_contexts"],
                            embed_layer.w, idx2word, top_k=10, n_samples=200)

    save_embeddings("embeddings_output", embed_layer.w, word2idx, idx2word, word_freq)


if __name__ == "__main__":
    main()