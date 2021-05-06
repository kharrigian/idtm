
"""
Transform raw social media text dataset into a bag of words representation
"""

###################
### Configuration
###################

## Date Parameters
MIN_DATE = "2017-01-01"
MAX_DATE = "2021-04-01"

## Directories
SUBREDDIT_DATA_DIR = f"./data/raw/depression/"
CACHE_DIR = "./data/processed/depression/"

## Data Type (submissions or comments)
DATA_TYPE = "comments"

## Vocabulary Parameters
MAX_N_GRAM = 3
PHRASE_THRESHOLD = 10
MIN_VOCAB_DF = 10
MIN_VOCAB_CF = 25
MAX_VOCAB_SIZE = 500000
RM_TOP_VOCAB = 250

## Script Meta Parameters
NUM_JOBS = 8
RANDOM_SEED = 42
SAMPLE_RATE = None
CACHE_TOP_K = 50
RERUN = True

###################
### Imports
###################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from collections import Counter

## External Library
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from gensim.models import Phrases
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import DictVectorizer

## Project Specific
from smtm.util.tokenizer import Tokenizer, STOPWORDS

###################
### Globals
###################

## Accounts to Ignore
IGNORABLES = set([
    "AutoModerator",
    "[deleted]",
    "[removed]"
])

## Dates
MIN_DATE = int(datetime.strptime(MIN_DATE, "%Y-%m-%d").timestamp())
MAX_DATE = int(datetime.strptime(MAX_DATE, "%Y-%m-%d").timestamp())

## Initialize Tokenizer
TOKENIZER = Tokenizer(stopwords=STOPWORDS,
                      keep_case=False,
                      negate_handling=True,
                      negate_token=False,
                      keep_punctuation=False,
                      keep_numbers=False,
                      expand_contractions=True,
                      keep_user_mentions=False,
                      keep_pronouns=True,
                      keep_url=False,
                      keep_hashtags=False,
                      keep_retweets=False,
                      emoji_handling=None,
                      strip_hashtag=False)

###################
### Helpers
###################

def load_data(filename,
              filters=None,
              min_date=None,
              max_date=None,
              exclude_ignorable_accounts=True,
              length_only=False):
    """

    """
    data = []
    length = 0
    with gzip.open(filename,"r") as the_file:
        for line_data in json.load(the_file):
            if exclude_ignorable_accounts and line_data.get("author") in IGNORABLES:
                continue
            if min_date is not None and line_data.get("created_utc") < min_date:
                continue
            if max_date is not None and line_data.get("created_utc") >= max_date:
                continue
            if length_only:
                length += 1
                continue
            if filters:
                line_data = dict((f, line_data.get(f,None)) for f in filters)
            data.append(line_data)
    if length_only:
        return length
    return data

class PostStream(object):

    """

    """

    def __init__(self,
                 filenames,
                 data_type,
                 min_date=None,
                 max_date=None,
                 stream_sentences=False,
                 exclude_ignorable_accounts=True,
                 return_metadata=False,
                 sample_rate=None,
                 random_state=42,
                 verbose=False):
        """

        """
        self.filenames = filenames
        self.min_date = min_date
        self.max_date = max_date
        self.stream_sentences = stream_sentences
        self.data_type = data_type
        if self.data_type == "submissions":
            self.text_fields = set(["selftext","title"])
        elif self.data_type == "comments":
            self.text_fields = set(["body"])
        else:
            raise ValueError("Expected data_type of 'submissions' or 'comments'")
        self.exclude_ignorable_accounts = exclude_ignorable_accounts
        self.return_metadata = return_metadata
        self.verbose = verbose
        self.random_state = random_state
        self.sample_rate = sample_rate
        self._initialize_filenames()
    
    def _initialize_filenames(self):
        """

        """
        filenames = []
        wrapper = lambda x: x
        if self.verbose:
            print("Isolating Nonempty Files")
            wrapper = lambda x: tqdm(x, total=len(x), desc="Filesize Filter", file=sys.stdout)
        for filename in wrapper(self.filenames):
            lf = load_data(filename,
                           min_date=self.min_date,
                           max_date=self.max_date,
                           exclude_ignorable_accounts=self.exclude_ignorable_accounts,
                           length_only=True)
            if lf > 0:
                filenames.append(filename)
        self.filenames = filenames
    
    def __len__(self):
        """

        """
        return len(self.filenames)

    def __iter__(self):
        """

        """
        sampler = np.random.RandomState(self.random_state)
        wrapper = lambda x: x
        if self.verbose:
            wrapper = lambda x: tqdm(x, desc="PostStream", file=sys.stdout)
        for filename in wrapper(self.filenames):
            file_data = load_data(filename,
                                  filters=["created_utc","id"]+list(self.text_fields),
                                  min_date=self.min_date,
                                  max_date=self.max_date,
                                  exclude_ignorable_accounts=self.exclude_ignorable_accounts)
            for post in file_data:
                if self.sample_rate is not None and sampler.rand() > self.sample_rate:
                    continue
                sentences = []
                if "title" in self.text_fields:
                    title_text = post.get("title")
                    if title_text is None:
                        title_text = ""
                    sentences.append(TOKENIZER.tokenize(post.get(title_text)))
                if "selftext" in self.text_fields:
                    selftext = post.get("selftext")
                    if selftext is None:
                        selftext = ""
                    for sentence in sent_tokenize(selftext):
                        sentences.append(TOKENIZER.tokenize(sentence))
                if "body" in self.text_fields:
                    bodytext = post.get("body")
                    if bodytext is None:
                        bodytext = ""
                    for sentence in sent_tokenize(bodytext):
                        sentences.append(TOKENIZER.tokenize(sentence))
                sentences = [s for s in sentences if len(s) > 0]
                if len(sentences) == 0:
                    continue
                if not self.stream_sentences:
                    sentences = [[i for s in sentences for i in s]]
                sentences = list(filter(lambda i: len(i) > 0, sentences))
                for tokens in sentences:
                    if self.return_metadata:
                        yield post.get("id"), post.get("created_utc"), tokens
                    else:
                        yield tokens

def learn_phrasers(filenames,
                   data_type,
                   verbose=False,
                   output_dir=None):
    """

    """
    ## Look for Existing Phrasers
    if output_dir is not None:
        if not RERUN:
            try:
                return load_phrasers(output_dir)
            except FileNotFoundError:
                pass
    ## Learn Vocabulary
    vocab_stream = PostStream(filenames,
                              min_date=MIN_DATE,
                              max_date=MAX_DATE,
                              stream_sentences=True,
                              data_type=data_type,
                              verbose=verbose,
                              random_state=RANDOM_SEED,
                              sample_rate=SAMPLE_RATE,
                              exclude_ignorable_accounts=True)
    print("Learning Initial Vocabulary (1-2 Grams)")
    ngrams = [2]
    phrasers =  [Phrases(sentences=vocab_stream,
                         max_vocab_size=MAX_VOCAB_SIZE,
                         threshold=PHRASE_THRESHOLD,
                         delimiter=" ")]
    current_n = 2
    while current_n < MAX_N_GRAM:
        print(f"Learning {current_n+1}-grams")
        phrasers.append(Phrases(phrasers[-1][vocab_stream],
                                max_vocab_size=MAX_VOCAB_SIZE,
                                threshold=PHRASE_THRESHOLD,
                                delimiter=" "))
        current_n += 1
        ngrams.append(current_n)
    print("Vocabulary Learning Complete")
    if output_dir is not None:
        _ = cache_phrasers(phrasers, ngrams, output_dir)
    return phrasers, ngrams

def cache_phrasers(phrasers,
                   ngrams,
                   output_dir):
    """

    """
    if not os.path.exists(f"{output_dir}/phrasers/"):
        _ = os.makedirs(f"{output_dir}/phrasers/")
    for phraser, ngram in zip(phrasers, ngrams):
        phraser_file = f"{output_dir}/phrasers/{ngram}.phraser"
        phraser.save(phraser_file)

def load_phrasers(output_dir):
    """

    """
    phraser_files = sorted(glob(f"{output_dir}/phrasers/*.phraser"))
    if len(phraser_files) == 0:
        raise FileNotFoundError("No phrasers found in the given model directory.")
    phrasers = []
    for pf in phraser_files:
        pf_ngram = int(os.path.basename(pf).split(".phraser")[0])
        pf_phraser = Phrases.load(pf)
        phrasers.append((pf_ngram, pf_phraser))
    phrasers = sorted(phrasers, key=lambda x: x[0])
    ngrams = [p[0] for p in phrasers]
    phrasers = [p[1] for p in phrasers]
    return phrasers, ngrams
    
def initialize_vectorizer(vocabulary):
    """
    Initialize a vectorizer that transforms a counter dictionary
    into a sparse vector of counts (with a uniform feature index)
    """
    ## Isolate Terms, Sort Alphanumerically
    ngram_to_idx = dict((t, i) for i, t in enumerate(vocabulary))
    ## Create Dict Vectorizer
    _count2vec = DictVectorizer(separator=":",dtype=int)
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def vectorize_data(filenames,
                   phrasers,
                   ngrams,
                   data_type,
                   verbose=True,
                   output_dir=None):
    """

    """
    ## Initialize Stream
    vector_stream = PostStream(filenames,
                               min_date=MIN_DATE,
                               max_date=MAX_DATE,
                               stream_sentences=True,
                               data_type=data_type,
                               verbose=verbose,
                               return_metadata=True,
                               random_state=RANDOM_SEED,
                               sample_rate=SAMPLE_RATE,
                               exclude_ignorable_accounts=True)
    ## Cache
    counts = {}
    times = {}
    for post_id, post_created, sentence in vector_stream:
        if post_id not in counts:
            counts[post_id] = Counter()
        if data_type == "comments":
            times[post_id] = post_created
        elif data_type == "submissions":
            if post_id not in times:
                times[post_id] = set()
            times[post_id].add(post_created)
        counts[post_id] += Counter([i for i in phrasers[0][sentence] if i.count(" ") == ngrams[0]-2])
        for n, p in zip(ngrams, phrasers):
            counts[post_id] += Counter([i for i in p[sentence] if i.count(" ") == n - 1])
    ## Get Vocabulary
    vocab = set()
    for v in tqdm(counts.values(), total=len(counts)):
        vocab.update(v.keys())
    vocab = sorted(vocab)
    ## Vectorize
    count2vec = initialize_vectorizer(vocab)
    post_ids = list(counts.keys())
    times = [times[p] for p in post_ids]
    X = sparse.vstack([count2vec.transform(counts[p]) for p in tqdm(post_ids,desc="Vectorization",file=sys.stdout)])
    ## Format Times
    if data_type == "submissions":
        times = [f"{min(t)} {max(t)}" for t in times]
    elif data_type == "comments":
        times = [f"{t} {t}" for t in times]
    ## Filter Vocabulary
    cf = X.sum(axis=0).A[0]
    df = (X!=0).sum(axis=0).A[0]
    top_k = set(np.argsort(cf)[-MAX_VOCAB_SIZE:-RM_TOP_VOCAB])
    vmask = np.logical_and(cf >= MIN_VOCAB_CF, df >= MIN_VOCAB_DF).nonzero()[0]
    vmask = list(filter(lambda v: v in top_k, vmask))
    vocab = [vocab[v] for v in vmask]
    X = X[:,vmask]
    print("Final Vocabulary Size: {}".format(X.shape[1]))
    ## Cache Document Term
    if output_dir is not None:
        _ = cache_document_term(X, post_ids, times, vocab, output_dir)
    return X, post_ids, times, vocab

def cache_document_term(X,
                        post_ids,
                        times,
                        vocabulary,
                        output_dir):
    """

    """
    ## Filenames
    X_filename = f"{output_dir}/data.npz"
    post_ids_filename = f"{output_dir}/posts.txt"
    times_filename = f"{output_dir}/times.txt"
    vocabulary_filename = f"{output_dir}/vocabulary.txt"
    ## Writing
    for obj, filename in zip([post_ids, times, vocabulary],
                             [post_ids_filename,times_filename,vocabulary_filename]):
        with open(filename,"w") as the_file:
            for item in obj:
                the_file.write(f"{item}\n")
    sparse.save_npz(X_filename, X)

def main():
    """

    """
    ## Update Output Directory
    OUTPUT_DIR = f"{CACHE_DIR}/{DATA_TYPE}/".replace("//","/")
    ## Establish Model Directory
    if not os.path.exists(OUTPUT_DIR):
        _ = os.makedirs(OUTPUT_DIR)
    ## Filenames
    filenames = sorted(glob(f"{SUBREDDIT_DATA_DIR}/{DATA_TYPE}/*.json.gz"))
    ## Try To Load Phrasers, Or Learn Them as Fallback
    phrasers, ngrams = learn_phrasers(filenames,
                                      data_type=DATA_TYPE,
                                      verbose=True,
                                      output_dir=OUTPUT_DIR)
    ## Get Vectorized Representation
    X, post_ids, times, vocabulary = vectorize_data(filenames=filenames,
                                                    data_type=DATA_TYPE,
                                                    phrasers=phrasers,
                                                    ngrams=ngrams,
                                                    verbose=True,
                                                    output_dir=OUTPUT_DIR)
    print("Script Complete!")

##################
### Execute
##################

if __name__ == "__main__":
    _ = main()