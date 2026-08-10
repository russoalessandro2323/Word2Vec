# Word2Vec from Scratch — Skip-gram in NumPy

Implementazione da zero (solo NumPy, nessun framework di deep learning) di un modello
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

## Roadmap dei prossimi miglioramenti

1. Subsampling delle parole ad alta frequenza durante la generazione delle coppie
2. Negative sampling, per rendere il training scalabile a vocabolari più grandi
3. Task di analogia vettoriale (es. "re" - "uomo" + "donna" ≈ "regina")
4. Visualizzazione 2D degli embedding (PCA / t-SNE)
5. Salvataggio e caricamento dei pesi allenati

## Utilizzo

Il codice è attualmente organizzato come notebook Jupyter. Le sezioni principali:

1. **Preprocessing**: `tokenize_text`, `build_vocab`, `encode`, `decode`
2. **Generazione dei dati di training**: `generate_training_pairs`
3. **Modello**: classi `Layer`, `EmbeddingLayer`, `DenseLayer`, `SoftMaxCrossEntropy`
4. **Training**: ciclo mini-batch con SGD, split train/test con shuffle preventivo
5. **Valutazione**: `most_similar_global`, `evaluate_topk_accuracy`

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

