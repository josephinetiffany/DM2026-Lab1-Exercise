"""Document-Term Matrix (DTM) construction tools for text analysis.

Provides tools to build and inspect document-term matrices from session data.
"""

from typing import Dict, List, Optional, Union
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    """Factory creating tools bound to the provided SessionState instance."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
    ) -> Dict[str, Union[str, int, float]]:
        """Builds a document-term matrix from session text data using CountVectorizer.

        Extracts token counts from document texts and stores the resulting sparse
        matrix, feature vocabulary, category labels, and fitted vectorizer in the
        session state.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams.
                Passed as a list of two ints, e.g. [1, 1] for unigrams or [1, 2]
                for unigrams + bigrams.
            max_df: Ignore terms with document frequency strictly higher than this threshold.
                If float in range [0.0, 1.0], represents proportion of documents.
                If int, represents absolute document count.
            min_df: Ignore terms with document frequency strictly lower than this threshold.
                If float in range [0.0, 1.0], represents proportion of documents.
                If int, represents absolute document count.
            max_features: Maximum vocabulary size ordered by term frequency.

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements,
            sparsity_pct, and labels_status (or error/status fields if execution fails).
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Load a dataset using load_dataset_tool first.",
            }
        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Loaded dataset in session does not contain a 'text' column.",
            }

        # Convert list representation back to tuple for sklearn vectorizer
        tuple_ngram = tuple(ngram_range) if isinstance(ngram_range, list) else ngram_range

        # Replace NaN/missing texts with empty string
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        try:
            vectorizer = CountVectorizer(
                ngram_range=tuple_ngram,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
            )
            dtm = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out().tolist()
        except ValueError as e:
            err_msg = str(e)
            if "empty vocabulary" in err_msg.lower():
                return {
                    "status": "error",
                    "message": (
                        "Vocabulary is empty after applying frequency thresholds. "
                        "Try lowering min_df, increasing max_df, or adjusting ngram_range."
                    ),
                }
            return {
                "status": "error",
                "message": f"Invalid vectorizer settings or parameters: {err_msg}",
            }

        # Store outputs in session
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        labels_status = "set"
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())
        else:
            labels_status = "skipped (category_name column not found)"

        # Compute matrix statistics & sparsity
        n_documents, n_terms = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_documents * n_terms
        sparsity_pct = (
            100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0
        )

        res_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": res_id,
            "status": "success",
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "labels_status": labels_status,
        }

        session.store_result(
            "build_dtm_tool",
            {
                "ngram_range": ngram_range,
                "max_df": max_df,
                "min_df": min_df,
                "max_features": max_features,
            },
            summary,
        )
        return summary

    return [build_dtm_tool]
