import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate check_missing_tool
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    assert result["status"] == "success"
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]  # Row index 7 is the last row with empty text

def test_check_missing_no_dataset():
    session = SessionState()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})
    assert result["status"] == "error"
    assert "No dataset loaded" in result["message"]

def test_check_duplicates_known_answer_3_drop_false():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture (8 rows)
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Test drop=False
    result = check_duplicates_tool.invoke({"drop": False})

    assert result["status"] == "success"
    assert result["duplicate_count"] == 1
    assert result["rows_dropped"] == 0
    assert result["total_rows_after"] == 8
    assert len(session.dataframe) == 8

def test_check_duplicates_known_answer_3_drop_true():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture (8 rows)
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Test drop=True
    result = check_duplicates_tool.invoke({"drop": True})

    assert result["status"] == "success"
    assert result["duplicate_count"] == 1
    assert result["rows_dropped"] == 2
    assert result["total_rows_after"] == 6
    assert len(session.dataframe) == 6

    # Verify that "always gamma delta" (rows 5 and 6 originally) was completely removed
    assert not (session.dataframe["text"] == "always gamma delta").any()

def test_check_duplicates_with_unigrams_list_column():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Add unigrams column containing unhashable list objects
    session.dataframe["unigrams"] = [["always"], ["alpha"], ["beta"], ["gamma"], ["delta"], ["always", "gamma", "delta"], ["always", "gamma", "delta"], []]

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Should not raise TypeError: unhashable type: 'list'
    result = check_duplicates_tool.invoke({"drop": False})

    assert result["status"] == "success"
    assert result["duplicate_count"] == 1
