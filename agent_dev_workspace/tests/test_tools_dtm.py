"""Tests for tools_dtm.py using agent_dev/sample_fixture.csv and TEST_FIXTURE.md known answers."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_known_answer_4():
    """Test build_dtm_tool against TEST_FIXTURE.md known answer #4."""
    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Get student build_dtm_tool bound to the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    # Assertions according to known answer #4:
    # 8 rows x 5 columns = 40 total elements, 21 non-zero, 47.5% sparsity
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5

    # Check terms in session feature_names
    expected_terms = ["alpha", "always", "beta", "delta", "gamma"]
    assert sorted(session.feature_names) == expected_terms

    # Check that vectorizer is saved in session.artifacts
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_build_dtm_no_dataset_loaded():
    """Test build_dtm_tool returns an error dict when no dataset is loaded in session."""
    session = SessionState()
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    assert res["status"] == "error"
    assert "No dataset loaded" in res["message"]


def test_term_frequency_known_answer_14():
    """Test term_frequency_tool against TEST_FIXTURE.md known answer #14."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture and build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_terms"] == 5

    # Check term ordering (frequency descending, then term alphabetically ascending for ties)
    expected_ordered_terms = ["always", "gamma", "alpha", "delta", "beta"]
    assert list(res["top_terms"].keys()) == expected_ordered_terms

    # Check counts match Known Answer #14
    expected_counts = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    assert res["top_terms"] == expected_counts


def test_term_frequency_no_dtm():
    """Test term_frequency_tool returns an error dict when feature_matrix is missing."""
    session = SessionState()
    tools = make_tools(session)
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    res = term_frequency_tool.invoke({})
    assert res["status"] == "error"
    assert "No document-term matrix found" in res["message"]
