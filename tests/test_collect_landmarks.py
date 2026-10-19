from pathlib import Path

import pytest

from scripts.collect_landmarks import select_image_records


def records(tmp_path):
    return [
        (tmp_path / "train" / "A" / f"a{i}.jpg", "A", "train")
        for i in range(5)
    ] + [
        (tmp_path / "test" / "A" / f"t{i}.jpg", "A", "test")
        for i in range(4)
    ] + [
        (tmp_path / "train" / "B" / f"b{i}.jpg", "B", "train")
        for i in range(3)
    ]


def test_sampling_is_deterministic_and_per_class_and_split(tmp_path):
    source = records(tmp_path)
    first = select_image_records(source, max_images_per_class=2, random_state=12)
    second = select_image_records(source, max_images_per_class=2, random_state=12)
    assert [item[0] for item in first] == [item[0] for item in second]
    assert len(first) == 6
    assert sum(label == "A" and split == "train" for _, label, split in first) == 2
    assert sum(label == "A" and split == "test" for _, label, split in first) == 2
    assert sum(label == "B" and split == "train" for _, label, split in first) == 2


def test_sampling_rejects_nonpositive_limit(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        select_image_records(records(tmp_path), max_images_per_class=0)


def test_unsplit_records_are_not_split_implicitly(tmp_path):
    source = [(tmp_path / "A" / f"a{i}.jpg", "A", "") for i in range(3)]
    assert select_image_records(source, max_images_per_class=2, random_state=4) == select_image_records(
        source, max_images_per_class=2, random_state=4
    )
