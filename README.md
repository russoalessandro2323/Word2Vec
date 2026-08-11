# Word2Vec from Scratch — Skip-gram in NumPy

Implementazione da zero di un modello
Word2Vec in variante Skip-gram, per l'apprendimento di word embedding a partire da
testo grezzo in lingua italiana.

## Caratteristiche

- **Preprocessing completo**: tokenizzazione con gestione esplicita dei confini di
  frase tramite un token speciale (`<NWL>`), costruzione del vocabolario con filtro
  per frequenza minima e gestione delle parole fuori vocabolario (`<UNK>`)
- **Architettura a layer componibili**, nello stile di un mini-framework: `EmbeddingLayer`,
  `DenseLayer`, `SoftMaxCrossEntropy`, ciascuno con forward/backward implementati
  manualmente — incluso il backward dell'embedding layer con accumulo corretto del
  gradiente su indici ripetuti all'interno dello stesso batch
- **Generazione delle coppie target-contesto** con finestra scorrevole configurabile,
  che rispetta i confini di frase (il contesto non attraversa mai `<NWL>`)
- **Training mini-batch con SGD**, validato su un sottoinsieme di Wikipedia in
  italiano (~350.000 parole grezze, vocabolario filtrato a diverse centinaia di parole)
- **Valutazione**:
  - qualitativa, tramite nearest neighbor via cosine similarity (`most_similar_global`)
  - quantitativa, tramite top-k accuracy su un test set held-out, con split
    train/test realizzato con shuffle preventivo per garantire rappresentatività

## Stato attuale

- Skip-gram con softmax completo su tutto il vocabolario (nessun negative sampling)
- Nessun subsampling delle stop word: le parole molto frequenti (articoli,
  preposizioni) dominano ancora sia il training sia la valutazione
- Testato su un corpus giocattolo scritto a mano e su un sottoinsieme di Wikipedia
  italiana (tramite la libreria `datasets` di HuggingFace)

## Dataset

Il modello è stato testato su:
- Un piccolo corpus tematico scritto a mano (per il debug e la validazione iniziale)
- Un sottoinsieme di [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
  (configurazione `20231101.it`), caricato tramite la libreria `datasets`

## Utilizzo

**Training end-to-end da riga di comando:**

```bash
pip install -r requirements.txt
python -m src.training
```

Scarica (o riusa, se già presente) il corpus da Wikipedia, allena il modello,
valuta il top-k accuracy sul test set e salva gli embedding in `embeddings_output/`.

