import re

import numpy as np


def tokenize_text(corpus):
    new_corpus = re.sub(r"[\n.!?]", " NWL ", corpus.lower())
    new_corpus = re.sub(r"[^\w\s]", " ", new_corpus)
    new_corpus = re.sub(r" NWL ", " <NWL> ", new_corpus)
    return new_corpus


def build_vocab(corpus, min_freq_cap=50):
    """Tokenizza il testo, conta le frequenze.
    Ritorna: word2idx, idx2word, word_freq
    """
    new_corpus = tokenize_text(corpus)
    words = new_corpus.split()

    n = len(words)
    # tengo solo le parole che compaiono almeno 1 volta ogni 1000 parole,
    # con un limite massimo (min_freq_cap) per non essere troppo aggressivi
    # su corpus molto grandi
    min_freq = int(n * 0.001)
    if min_freq > min_freq_cap:
        min_freq = min_freq_cap

    unique_words, frequences = np.unique(words, return_counts=True)

    filtered_words = unique_words[frequences >= min_freq]
    filtered_freqs = frequences[frequences >= min_freq]

    word2idx = {}
    idx2word = {}
    word_freq = {}

    nwl_idx = np.where(filtered_words == "<NWL>")[0]
    unk_idx = np.where(frequences < min_freq)[0]
    nwl_frq = filtered_freqs[nwl_idx]
    unk_frq = np.sum(frequences[unk_idx])

    word_freq["<NWL>"] = nwl_frq[0]
    word2idx["<NWL>"] = 0
    idx2word[0] = "<NWL>"
    filtered_words = np.delete(filtered_words, nwl_idx, 0)
    filtered_freqs = np.delete(filtered_freqs, nwl_idx, 0)

    for idx, word in enumerate(filtered_words):
        word2idx[str(word)] = idx + 1
        idx2word[idx + 1] = str(word)
        word_freq[str(word)] = filtered_freqs[idx]

    last_idx = len(filtered_words) + 1
    word_freq["<UNK>"] = unk_frq
    word2idx["<UNK>"] = last_idx
    idx2word[last_idx] = "<UNK>"

    return word2idx, idx2word, word_freq


def encode(corpus, word2idx):
    out = []
    new_corpus = tokenize_text(corpus)
    words = new_corpus.split()

    for word in words:
        try:
            out.append(word2idx[word])
        except KeyError:
            out.append(word2idx["<UNK>"])
    return out


def decode(corpus_indexes, idx2word):
    return [idx2word[idx] for idx in corpus_indexes]


def generate_training_pairs(corpus_indices, window_size=3):
    """Per ogni parola nel corpus, genera coppie (target, context)
    scorrendo una finestra di +-window_size. <NWL> agisce da muro
    simmetrico: il contesto non attraversa mai un confine di frase.

    Ritorna: lista di coppie [target_idx, context_idx]
    """
    out = []
    last_idx = len(corpus_indices) - 1

    for pos, idx in enumerate(corpus_indices):
        if idx == 0:  # non generiamo coppie con <NWL> come target
            continue

        for j in range(1, window_size + 1):  # sinistra
            if (pos - j) >= 0:
                out.append([idx, corpus_indices[pos - j]])
                if corpus_indices[pos - j] == 0:
                    break

        for j in range(1, window_size + 1):  # destra
            if (pos + j) <= last_idx:
                out.append([idx, corpus_indices[pos + j]])
                if corpus_indices[pos + j] == 0:
                    break

    return out