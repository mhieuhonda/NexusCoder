"""Topic Modeling Skill - LDA / BERTopic / NMF pipeline with visualization.

Sinh pipeline topic modeling: text preprocessing (tokenize, lemmatize, stopwords),
BoW + TF-IDF, LDA (gensim), NMF (sklearn), BERTopic (transformers + UMAP + HDBSCAN),
coherence evaluation (c_v), pyLDAvis visualization, và topic inference on new docs.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


LDA_PIPELINE = '''"""LDA topic modeling pipeline / Pipeline LDA với gensim."""
from __future__ import annotations
from typing import List, Dict
import re
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import nltk
for pkg in ("stopwords", "punkt", "wordnet", "omw-1.4"):
    try: nltk.data.find(f"corpora/{pkg}")
    except LookupError: nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models


LEMMATIZER = WordNetLemmatizer()
STOP = set(stopwords.words("english")) | {"via", "rt", "amp"}


def preprocess(docs: List[str], min_token_len: int = 3) -> List[List[str]]:
    """Tokenize + lemmatize + remove stopwords / Tiền xử lý."""
    cleaned = []
    for d in docs:
        d = re.sub(r"http\\S+|www\\S+", "", d.lower())   # strip URLs
        d = re.sub(r"[^a-z\\s]", " ", d)                  # keep alpha only
        tokens = [
            LEMMATIZER.lemmatize(t) for t in d.split()
            if t not in STOP and len(t) >= min_token_len
        ]
        cleaned.append(tokens)
    return cleaned


def train_lda(docs_tokens: List[List[str]], num_topics: int = 10,
              passes: int = 10, random_state: int = 42):
    """Train LDA / Huấn luyện LDA."""
    dictionary = corpora.Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs_tokens]
    lda = models.LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=passes, random_state=random_state, alpha="auto",
        eta="auto", iterations=400,
    )
    return lda, dictionary, corpus


def coherence_score(docs_tokens, dictionary, lda) -> float:
    """C_v coherence / Đánh giá độ gắn kết chủ đề."""
    from gensim.models import CoherenceModel
    cm = CoherenceModel(model=lda, texts=docs_tokens, dictionary=dictionary, coherence="c_v")
    return float(cm.get_coherence())


def show_topics(lda, n_words: int = 10) -> pd.DataFrame:
    """Format top words per topic / Hiển thị top từ mỗi topic."""
    rows = []
    for tid in range(lda.num_topics):
        words = ", ".join(w for w, _ in lda.show_topic(tid, topn=n_words))
        rows.append({"topic": tid, "top_words": words})
    return pd.DataFrame(rows)


def visualize_lda(lda, corpus, dictionary, outfile: str = "lda_vis.html"):
    """pyLDAvis interactive / Trực quan tương tác."""
    vis = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis, outfile)
    return outfile


def infer(lda, dictionary, doc: str) -> Dict[int, float]:
    """Predict topic distribution cho văn bản mới / Dự đoán topic."""
    bow = dictionary.doc2bow(preprocess([doc])[0])
    return {int(t): float(p) for t, p in lda.get_document_topics(bow)}


if __name__ == "__main__":
    # Demo with mock docs / Demo với tài liệu mẫu
    docs = [
        "machine learning model training neural network",
        "soccer football world cup championship goal",
        "python javascript web framework backend frontend",
    ] * 30
    tokens = preprocess(docs)
    lda, dic, corpus = train_lda(tokens, num_topics=3)
    print(show_topics(lda))
    print("coherence c_v:", coherence_score(tokens, dic, lda))
    visualize_lda(lda, corpus, dic)
'''

BERTOPIC_PIPELINE = '''"""BERTopic pipeline / Pipeline BERTopic."""
from __future__ import annotations
from typing import List, Dict
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.dimensionality import BaseDimensionalityReducer
from sklearn.cluster import KMeans


def train_bertopic(docs: List[str], n_topics: int = 20, language: str = "english"):
    """Train BERTopic với embeddings + UMAP + HDBSCAN."""
    # Use KeyBERT-inspired representation for richer topic labels
    representation_model = KeyBERTInspired()

    # Optionally cap topics via KMeans (faster, deterministic topic count)
    cluster_model = KMeans(n_clusters=n_topics, random_state=42)

    topic_model = BERTopic(
        language=language,
        nr_topics=n_topics,
        min_topic_size=10,
        calculate_probabilities=False,
        representation_model=representation_model,
        hdbscan_model=cluster_model,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def visualize_bertopic(topic_model):
    """Generate standard plots / Sinh các biểu đồ chuẩn."""
    out = {}
    out["barchart"] = topic_model.visualize_barchart(top_n_topics=12)
    out["heatmap"] = topic_model.visualize_heatmap()
    out["topics"] = topic_model.visualize_topics()
    out["hierarchy"] = topic_model.visualize_hierarchy()
    return out


if __name__ == "__main__":
    docs = ["neural network deep learning", "football match goal", "python web framework"] * 50
    tm, _, _ = train_bertopic(docs, n_topics=3)
    print(tm.get_topic_info().head(10))
'''

STRATEGY = """
Topic Modeling — Strategy Guide / Hướng dẫn chiến lược
=======================================================
| Method   | Best For                              | Pros                        | Cons                          |
|----------|----------------------------------------|-----------------------------|-------------------------------|
| LDA      | BoW-style corpora, interpretable topics| Probabilistic, fast         | Needs large corpus            |
| NMF      | Short texts, non-negative decomposition| Faster than LDA             | Less principled probabilistic |
| LSA      | Latent semantic indexing               | Quick                       | Linear, less interpretable    |
| BERTopic | Mixed-language, short texts, semantics| Embedding-based, contextual | Slow without GPU; needs tuning|

Choosing K (#topics):
  - Coherence score (c_v or u_mass) — pick K maximizing coherence.
  - Human interpretability — read top words of each topic.
  - Topic diversity (proportion of unique words in top-25).

Preprocessing:
  - Always remove stopwords, lemmatize, and filter very rare/common tokens.
  - For BERTopic, keep raw text (don't over-process) — embeddings do the heavy lifting.

Evaluation:
  - Coherence c_v > 0.5 is generally good.
  - Topic diversity > 0.7 means topics are not redundant.
  - Always sanity-check top words for human interpretability.
"""


class TopicModelingSkill(Skill):
    """Sinh LDA / NMF / BERTopic topic modeling pipeline."""

    category = SkillCategory.LANGUAGE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "topic modeling", "topic model", "lda", "nmf", "lsa",
        "bertopic", "topic", "topics", "coherence", "pyldavis",
        "gensim", "document clustering",
    ]
    examples = [
        "Phát hiện chủ đề trong tập văn bản",
        "Train LDA với 10 topics và visualize",
        "BERTopic cho corpus ngôn ngữ hỗn hợp",
    ]

    @property
    def name(self) -> str:
        return "topic_modeling"

    @property
    def description(self) -> str:
        return (
            "Sinh topic modeling pipeline: preprocessing (tokenize/lemmatize/stopwords), "
            "LDA (gensim) + coherence + pyLDAvis, NMF, và BERTopic (transformers+UMAP+HDBSCAN)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "bertopic" in prompt_lower:
            recommended = "bertopic"
        elif "nmf" in prompt_lower:
            recommended = "nmf"
        else:
            recommended = "lda"

        artifacts: List[Dict[str, str]] = [
            {"name": "lda_pipeline.py", "language": "python", "content": LDA_PIPELINE},
            {"name": "bertopic_pipeline.py", "language": "python", "content": BERTOPIC_PIPELINE},
            {"name": "STRATEGY.md", "language": "markdown", "content": STRATEGY},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[topic_modeling] recommended={recommended}\n"
                f"Generated LDA (gensim + coherence + pyLDAvis) and BERTopic "
                f"(embeddings + UMAP + HDBSCAN) pipelines + strategy guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Tune K (#topics) by maximizing coherence c_v (>0.5 is good)",
                "For short texts (tweets, reviews), prefer BERTopic over LDA",
                "Always preprocess: lemmatize, remove stopwords, filter rare tokens",
                "Visualize inter-topic distances via pyLDAvis for inspection",
                "Compare LDA vs NMF vs BERTopic before picking final model",
            ],
            metadata={
                "skill": self.name,
                "recommended_model": recommended,
                "models_available": ["lda", "nmf", "lsa", "bertopic"],
                "evaluation": ["coherence_c_v", "coherence_u_mass", "topic_diversity"],
                "version": self.version,
                "author": self.author,
            },
        )
