"""Phase 4: figure detection and the derived status."""

from __future__ import annotations


def test_all_captioned_figures_found(extracted):
    assert len(extracted["figures"]) == 4
    assert {f["figure_number"] for f in extracted["figures"]} == {"1", "2", "3", "5"}


def test_all_figures_ok(extracted):
    assert all(f["status"] == "OK" for f in extracted["figures"])


def test_page_frame_not_a_figure(extracted):
    """The fixture draws a border around every page; it must not become a figure."""
    for f in extracted["figures"]:
        w = f["bbox"][2] - f["bbox"][0]
        h = f["bbox"][3] - f["bbox"][1]
        assert w < 500 and h < 800, f["figure_id"]


def test_png_exists_and_is_valid(extracted):
    from pathlib import Path

    from PIL import Image

    base = extracted["dir"]
    for f in extracted["figures"]:
        assert f["png_path"], f["figure_id"]
        path = base / f["png_path"]
        assert path.exists(), path
        with Image.open(path) as im:
            assert im.size[0] > 100 and im.size[1] > 50


def test_caption_included_in_crop(extracted):
    """The crop must start at or above the caption so the image is self-describing."""
    for f in extracted["figures"]:
        assert f["bbox"][1] <= f["graphic_bbox"][1]


def test_no_figure_inside_a_table(extracted):
    assert all(f["is_inside_table"] is False for f in extracted["figures"])


def test_status_field_is_derived_not_borrowed(extracted):
    """Every record carries the locally-derived status, not a tool-supplied one."""
    for f in extracted["figures"]:
        assert f["status"] in ("OK", "SUSPECT", "FAILED")
        assert "detector" in f


def test_detector_is_primary_for_codebooks(extracted):
    """Caption-anchored detection is the primary path; pdffigures2 supplements it."""
    assert all(f["detector"] == "pycaption" for f in extracted["figures"])