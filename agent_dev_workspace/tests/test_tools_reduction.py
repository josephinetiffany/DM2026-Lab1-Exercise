import pytest
import numpy as np
import matplotlib.pyplot as plt
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def get_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture (8 documents)
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_pca_known_answer_7():
    session = get_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert res["status"] == "success"
    assert res["method"] == "pca"
    assert res["coordinates_shape"] == [8, 2]
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)

    # Check top 3 coordinates against Known Answer #7
    expected_top_3 = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587]
    ]
    assert np.allclose(res["top_coordinates"][:3], expected_top_3, atol=1e-3)
    assert session.pending_figure is not None

def test_tsne_known_answer_8():
    session = get_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    assert res["status"] == "success"
    assert res["method"] == "tsne"
    assert res["coordinates_shape"] == [8, 2]
    assert session.pending_figure is not None

    # Retrieve full coordinates array from artifacts
    coords = session.artifacts["reduced_tsne"]
    assert coords.shape == (8, 2)

    # Verify that duplicate documents (rows 5 and 6) land on the exact same coordinate
    assert np.allclose(coords[5], coords[6], atol=1e-3)

    # Compute centroids for catA (docs 0,1,2) and catB (docs 3,4,5,6)
    catA_coords = coords[0:3]
    catB_coords = coords[3:7]

    catA_centroid = np.mean(catA_coords, axis=0)
    catB_centroid = np.mean(catB_coords, axis=0)

    # Verify catA docs (0-2) are closer to catA_centroid than catB_centroid
    for doc_coord in catA_coords:
        dist_to_A = np.linalg.norm(doc_coord - catA_centroid)
        dist_to_B = np.linalg.norm(doc_coord - catB_centroid)
        assert dist_to_A < dist_to_B

    # Verify catB docs (3-6) are closer to catB_centroid than catA_centroid
    for doc_coord in catB_coords:
        dist_to_A = np.linalg.norm(doc_coord - catA_centroid)
        dist_to_B = np.linalg.norm(doc_coord - catB_centroid)
        assert dist_to_B < dist_to_A

def test_umap_known_answer_9():
    session = get_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    assert res["status"] == "success"
    assert res["method"] == "umap"
    assert res["coordinates_shape"] == [8, 2]

    expected_top_3 = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712]
    ]
    assert np.allclose(res["top_coordinates"][:3], expected_top_3, atol=1e-1)
    assert session.pending_figure is not None
