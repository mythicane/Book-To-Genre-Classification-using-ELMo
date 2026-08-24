# Book To Genre using ML

Classifies full book texts into their associated genre(s) using classical/early
NLP techniques (TF-IDF extractive summarization + [ELMo](https://en.wikipedia.org/wiki/ELMo)
embeddings + a multi-label classifier). Built for **StoryForge**, to
auto-suggest genres given a manuscript.

Training/reference text comes in part from the [Project Gutenberg](https://www.gutenberg.org/)
dataset (public-domain books), alongside the Brown corpus and the Kaggle
"10,000 Books and Their Genres" dataset linked below.

This is a research pipeline, not a packaged CLI: each script exposes functions
you call by editing the `if __name__ == "__main__":` block at the bottom (most
of the useful calls are already there, just commented out). This README walks
through what each file does and the order you'd run them in.

## Pipeline overview

```
textcompressor.py  →  elmo.py  →  databuilder.py  →  genreclassifier.py
  (shrinks a book      (embeds       (builds the        (trains/runs the
   into its most        text via     training dataset    multi-label genre
   "important"          ELMo)        from Kaggle CSV +   classifier)
   sentences)                        clean_books_and_
                                      genres.p)
```

`Main.py` is a minimal entry point that just runs `textcompressor.run()` on a
book file and prints the compressed text — useful for sanity-checking the
compressor in isolation.

## Setup

Create an environment (conda or venv) with:

```bash
pip install nltk scikit-learn tensorflow tensorflow_hub pandas numpy tqdm matplotlib
```

Then download the NLTK corpora these scripts pull from:

```python
import nltk
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')  # newer NLTK versions split this out separately
```

`elmo.py` downloads the ELMo model itself from TensorFlow Hub
(`https://tfhub.dev/google/elmo/3`) the first time `elmo_init()` runs, so the
first call will be slow and needs an internet connection.

## 1. `textcompressor.py` — extractive summarization

Given a long text, scores each sentence with TF-IDF (term frequency × inverse
document frequency) and keeps only the sentences that score above a threshold
scaled by a `HANDICAP` constant — the higher the handicap, the more selective
(shorter) the output. This is compression, not generative summarization: every
sentence it keeps is copied verbatim from the input.

Three ways to run it, all currently commented out at the bottom of the file:

- `demo(HANDICAP)` — interactive terminal demo; prompts you to pick a sample
  text (fan drafts included in this repo, or Brown-corpus excerpts) and prints
  the before/after.
- `run_personal_demo(HANDICAP, text)` — single-pass compression on your own
  text string, no parallelism, handicap of your choosing.
- `run(text)` — the production path used by the rest of the pipeline. Splits
  the text into 2,000-word chunks, compresses each chunk in parallel
  (`multiprocessing.Pool`) with a fixed handicap of `1.2`, then recursively
  re-compresses the combined output until it's under 2,000 words.

Run standalone: uncomment your choice of the three calls at the bottom of
`textcompressor.py` and run `python textcompressor.py`.

The only sample manuscript included in this repo is `The_Great_Gatsby.txt`.
`demo(HANDICAP)`'s interactive "fan-made literature" branch (`[f]`) references
a few personal drafts that aren't checked into this repo, so those menu
options won't find a file — use the "classical literature" branch (`[c]`,
Brown-corpus excerpts) or `run_personal_demo()`/`run()` with your own text
instead.

## 2. `elmo.py` — embeddings

Wraps a frozen ELMo model (a bi-directional LSTM language model, chosen over
BERT/GPT-family models because of data-sourcing/consent concerns with their
training corpora — see the module docstring for the full reasoning) from
TensorFlow Hub.

- `elmo_init()` — loads and returns the ELMo object.
- `elmo_embed(elmo, s)` — given the ELMo object and a string, returns the
  embedding.
- `demo()` — embeds a handful of Brown-corpus excerpts and prints their
  lengths/embeddings side by side.

Run standalone with `python elmo.py` to sanity-check that ELMo loads and can
embed a "Hello World!" string.

## 3. `databuilder.py` — building the training dataset

Builds the labeled dataset that `genreclassifier.py` trains on, from the
Kaggle dataset ["10,000 Books and Their Genres (standardized)"](https://www.kaggle.com/datasets/michaelrussell4/10000-books-and-their-genres-standardized).

`clean_books_and_genres.p` (a pre-cleaned pickle of that dataset) is already
in this repo. You additionally need `books_and_genres.csv` — download it from
the Kaggle link above and place it in the repo root; it's not checked in here.

Two functions, meant to be run once each, in order:

1. **`load_dataframe()`** — merges the pickled clean data with the raw CSV's
   text column, drops unneeded columns, and writes `books_and_genres_cleaned.csv`.
2. **`add_embeddings()`** — reads `books_and_genres_cleaned.csv`, runs each
   book's text through `textcompressor.run()` then `elmo_embed()`, and saves
   the result (including a new `elmo_embeddings` column) as `data.pkl`. This
   is the slow step — one ELMo embedding per book.

Uncomment whichever function you need at the bottom of the file and run
`python databuilder.py`. `add_embeddings()` writes its output to `data.pkl`,
which `genreclassifier.get_data()` reads directly.

## 4. `genreclassifier.py` — training and prediction

Trains a multi-label genre classifier on the ELMo embeddings built above, and
lets you run predictions on new text.

- **`get_data()`** — loads the embeddings pickle, multi-label-binarizes the
  genre lists with `MultiLabelBinarizer` (saved to `mlb.pkl`), and pads all
  embeddings to the same length.
- **`reduce_dimensions(X)`** — standard-scales and PCA-reduces the (very high
  dimensional) ELMo embeddings down to 191 components, saving the fitted PCA
  to `pca_model.pkl`. This is necessary because raw ELMo embeddings are far
  wider than the number of training examples.
- **`visualize_genres(y)`** — bar chart of how many examples exist per genre
  (useful for spotting class imbalance).
- **`build_logistic()`** — trains a One-vs-Rest `LogisticRegression` over the
  reduced embeddings, prints a `classification_report`, and pickles the model
  to `logisticmodel.pkl`. (The module also imports `SVC` for an alternative
  One-vs-Rest SVM approach, though only the logistic path is wired up at the
  bottom of the file.)
- **`predict(text, model_name)`** — the inference entry point. Loads the
  trained model (`"logistic"` or `"svc"`, provided its `.pkl` exists), the
  `MultiLabelBinarizer`, and the PCA model; compresses and embeds your input
  text the same way training data was processed; and prints a probability per
  genre.

Typical run order (uncomment the relevant lines at the bottom of the file):

1. `build_logistic()` once, to train and save the model + PCA + label
   binarizer.
2. Point `text = open(<your file>, 'r').read()` at whatever manuscript you
   want classified, set `model_name = "logistic"`, and run
   `python genreclassifier.py` to get per-genre probabilities.

`predict()` can be pointed at `The_Great_Gatsby.txt` (the sample included in
this repo) or any manuscript of your own.

## `Main.py` — quick compressor smoke test

```bash
python Main.py
```

Reads a book text file (edit the hardcoded filename near the bottom of the
file to point at one that exists — several alternatives are left commented
out) and prints the `textcompressor.run()` output. Good for confirming your
environment is set up correctly before running the heavier ELMo/classifier
steps above.

## Notes

- All file paths in these scripts are relative and hardcoded — run everything
  from the repo root, and expect to hand-edit a filename here and there
  (that's the normal way to point these scripts at a different book).
- `genreclassifier.build_logistic()` and `databuilder.add_embeddings()` are
  both slow (ELMo embedding + PCA over a ~1,300-book dataset); budget time
  accordingly, especially without a GPU.
