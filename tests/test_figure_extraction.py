"""Phase 4: figure detection, independent verification, and derived status."""

from __future__ import annotations

from src.extraction.figure_extractor import (
    CONF_CAPTION,
    CONF_FIG_DEGRADED,
    CONF_MULTI,
    CONF_VECTOR,
    FIG_OK,
)


def test_all_captioned_figures_found(extracted):
    assert len(extracted["figures"]) == 4
    assert {f["figure_number"] for f in extracted["figures"]} == {"1", "2", "3", "5"}


def test_all_figures_ok(extracted):
    assert all(f["status"] == FIG_OK for f in extracted["figures"])


def test_page_frame_not_a_figure(extracted):
    """The fixture draws a border around every page; it must not become a figure."""
    for f in extracted["figures"]:
        w = f["bbox"][2] - f["bbox"][0]
        h = f["bbox"][3] - f["bbox"][1]
        assert w < 500 and h < 800, f["figure_id"]


def test_png_exists_and_is_valid(extracted):
    from PIL import Image

    base = extracted["dir"]
    for f in extracted["figures"]:
        assert f["png_path"], f["figure_id"]
        path = base / f["png_path"]
        assert path.exists(), path
        with Image.open(path) as im:
            assert im.size[0] > 100 and im.size[1] > 50


def test_caption_included_in_crop(extracted):
    for f in extracted["figures"]:
        assert f["bbox"][1] <= f["graphic_bbox"][1]


def test_no_figure_inside_a_table(extracted):
    assert all(f["is_inside_table"] is False for f in extracted["figures"])


def test_confidence_labels_are_from_the_spec_vocabulary(extracted):
    allowed = {CONF_MULTI, CONF_CAPTION, CONF_VECTOR, CONF_FIG_DEGRADED}
    for f in extracted["figures"]:
        assert f["confidence"] in allowed, (f["figure_id"], f["confidence"])
        assert f["status"] in ("OK", "SUSPECT", "FAILED")


def test_multi_signal_requires_both_detectors(extracted):
    """`multi_signal` may only be claimed when PDFFigures 2.0 also reported it."""
    for f in extracted["figures"]:
        if f["confidence"] == CONF_MULTI:
            assert "pdffigures2" in f.get("signals", []), f["figure_id"]


def test_pdffigures2_crops_are_persisted(extracted):
    """p2 renders live in a temp dir that is deleted; they must be copied out."""
    base = extracted["dir"]
    for f in extracted["figures"]:
        rel = f.get("pdffigures2_png")
        if rel:
            assert (base / rel).exists(), rel


def test_signals_are_recorded(extracted):
    for f in extracted["figures"]:
        assert f.get("signals"), f["figure_id"]
        assert "vector_region" in f["signals"]


def test_vector_only_records_have_no_caption(extracted):
    for f in extracted["figures"]:
        if f["confidence"] == CONF_VECTOR:
            assert not f.get("caption")
