# !pip install python-terrier
# !pip install fastrank
# !pip install lightgbm
# !pip install googletrans==4.0.0rc1
# !sudo apt-get install swig
# !sudo pip install jamspell
# !wget https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
# !tar -xvf en.tar.gz

from pyterrier.measures import *
import pandas as pd
import pyterrier as pt
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
# import fastrank
# import lightgbm as lgb
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from googletrans import Translator
# import jamspell


def rf():
    df = pd.read_csv('data/df.csv', low_memory=False)
    docs_df = pd.read_csv('data/docs_df.csv', low_memory=False)
    topics = pd.read_csv("data/Annotation - Query.csv", low_memory=False)
    qrels = pd.read_csv("data/Annotation - Evaluation.csv", low_memory=False)
    tr_va_topics, test_topics = train_test_split(
        topics, test_size=0.3, random_state=42)
    train_topics, valid_topics = train_test_split(
        tr_va_topics, test_size=0.1, random_state=42)

    # jsp = jamspell.TSpellCorrector()
    # assert jsp.LoadLangModel('data/en.bin')

    nltk.download('stopwords')
    stops = set(stopwords.words('english'))
    translator = Translator()

    def _query_rewrite(q):
        query = q["query"]
        # Translation
        if not query.isascii():
            query = translator.translate(query).text

        # # Spell corrector
        # query = jsp.FixFragment(query)

        # Remove stop word
        terms = query.split(" ")
        terms = [t for t in terms if not t in stops]
        terms = [term.lower() for term in terms]
        query = " ".join(terms)
        return query

    if not pt.started():
        pt.init()

    index_dir = './metdocs_index'
    indexer = pt.DFIndexer(index_dir, overwrite=True)
    index_ref = indexer.index(docs_df["text"], docs_df["docno"])
    index = pt.IndexFactory.of(index_ref)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")

    ltr_feats = pt.apply.query(_query_rewrite) >> pt.BatchRetrieve(index) >> (
        bm25
        **
        tfidf
        **  # abstract coordinate match
        pt.BatchRetrieve(index, wmodel="CoordinateMatch")
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=400, verbose=1, random_state=42, n_jobs=2)
    rf_pipe = ltr_feats >> pt.ltr.apply_learned_model(rf)
    rf_pipe.fit(train_topics, qrels)
    return rf_pipe


def merge(search_results):
    df = pd.read_csv('data/df.csv', low_memory=False)
    docs = search_results.merge(df, how='left', on='docno')
    return docs

# if __name__ == '__main__':
#     model = rf()
#     print("Model trained.")

#     query = "Who was influenced by Claude Monet"
#     search = model.search(query).head(20)
#     print(search[['docno']])
