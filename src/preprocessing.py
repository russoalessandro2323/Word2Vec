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


def generate_training_pairs(corpus_indexes, idx2word, word_probs, window_size=3):
    # Per ogni parola nel corpus, genera coppie (target, context)
    # scorrendo una finestra di ±window_size

    # Output atteso: lista di coppie (target_idx, context_idx)
    # Attenzione ai bordi delle frasi/testo (non puoi prendere contesto "fuori" dal testo)

    filtered_indexes = subsampling(corpus_indexes, idx2word, word_probs)
    last_idx = len(filtered_indexes) - 1
    out = []

    for pos, idx in enumerate(filtered_indexes):

      if idx != 0:
        for j in range(1, window_size + 1): # sx
            if (pos - j) >= 0:
              out.append([idx, filtered_indexes[pos - j]])
              if filtered_indexes[pos - j] == 0: # if context == '<NWL>' break the cicle for that word
                break

        for j in range(1, window_size + 1): # dx
          if (pos + j) <= last_idx:
            out.append([idx, filtered_indexes[pos + j]])

            if filtered_indexes[pos + j] == 0: # if context == '<NWL>' break the cicle for that word
              break

    return out

def subsampling(corpus_indexes, idx2word, word_probs, t=1e-3):
  keep_probs = {}
  for idx, word in idx2word.items():
      if idx == 0:  # <NWL> mantenuto sempre
          keep_probs[idx] = 1.0
          continue

      f = word_probs.get(word, 1e-6)
      keep_probs[idx] = np.sqrt(t / f) if f > t else 1.0

  filtered_indices = []
  for idx in corpus_indexes:
      if idx == 0:
          filtered_indices.append(idx)
      elif np.random.rand() < keep_probs.get(idx, 1.0):
          filtered_indices.append(idx)

  return filtered_indices