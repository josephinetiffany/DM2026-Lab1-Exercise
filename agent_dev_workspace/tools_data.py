import pandas as pd
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def check_missing_tool() -> dict:
        """Checks the loaded dataframe for missing, empty, or whitespace-only text entries.

        Inspects session.dataframe['text'] to identify rows that are null (NaN/None),
        empty strings, or strings containing only whitespace.

        Returns:
            dict: Summary containing result_id, total_rows, missing_count, missing_indices (top 10),
                  and an optional note if indices were truncated.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }
        
        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Column 'text' not found in session.dataframe."
            }

        df = session.dataframe
        text_series = df["text"]

        # Check for NaN/None OR empty string after stripping whitespace
        is_missing = text_series.isna() | text_series.astype(str).str.strip().eq("")
        
        full_missing_indices = df.index[is_missing].tolist()
        missing_count = len(full_missing_indices)
        total_rows = len(df)

        result_id = session.next_result_id("check_missing")

        # Keep summary small for JSON serializability (top 10 indices)
        summary_indices = full_missing_indices[:10]

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": summary_indices
        }

        if missing_count > 10:
            summary["note"] = f"Showing first 10 of {missing_count} missing indices. Full list saved in full_report."

        # Store complete list of missing indices in full_report
        full_report = {
            "missing_indices": full_missing_indices,
            "total_rows": total_rows,
            "missing_count": missing_count
        }

        session.store_result("check_missing_tool", {}, summary, full_report=full_report)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        Matching Master's X.drop_duplicates(keep=False) behavior, when drop=True,
        every copy of a duplicated row is removed (not just the extra copy).

        Args:
            drop (bool): If True, drops all copies of duplicate rows and updates session.dataframe in place. Default is False.

        Returns:
            dict: Summary containing result_id, total_rows_before, duplicate_count, rows_dropped, total_rows_after, and dropped status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first."
            }

        df = session.dataframe
        total_rows_before = len(df)

        # Exclude unhashable columns (like list/unigrams) to prevent TypeError if tokenize_tool ran first
        subset_cols = [c for c in df.columns if not df[c].apply(lambda x: isinstance(x, (list, dict))).any()]

        # Standard pandas duplicated check (counts extra copies, matching Known Answer #3)
        duplicate_count = int(df.duplicated(subset=subset_cols).sum())

        if drop and duplicate_count > 0:
            # Drop every copy of a duplicated row, matching Master's drop_duplicates(keep=False)
            cleaned_df = df.drop_duplicates(subset=subset_cols, keep=False).reset_index(drop=True)
            session.dataframe = cleaned_df
            rows_dropped = total_rows_before - len(cleaned_df)
            dropped = True
        else:
            rows_dropped = 0
            dropped = False

        total_rows_after = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_before": total_rows_before,
            "duplicate_count": duplicate_count,
            "rows_dropped": rows_dropped,
            "total_rows_after": total_rows_after,
            "dropped": dropped
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    return [check_missing_tool, check_duplicates_tool]
