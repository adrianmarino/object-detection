from lib.pascal_voc.bounding_box import BoundingBox
from lib.pascal_voc.dataset import Dataset
from lib.pascal_voc.image_size import Image


def test_get_samples():
    # Prepare
    dataset = Dataset('./dataset1')

    # Perform
    samples = list(dataset.samples())

    # Asserts
    assert len(samples) == 1
    assert samples[0].image == \
           Image('/home/adrian/development/machine-learning/dataset/IMG_20200219_204537.jpg', 4608, 2112, 3)
    assert len(samples[0].bounding_boxes) == 1
    assert samples[0].bounding_boxes[0] == BoundingBox('joker', 727, 136, 1552, 873)


def test_group_bounding_boxes_by():
    # Prepare
    dataset = Dataset('./dataset2')

    # Perform
    groups = dataset.group_bounding_boxes_by('class_name')

    # Asserts
    assert len(groups) == 2

    assert len(groups['joker']) == 1
    assert any(b.class_name == 'joker' for b in groups['joker']), True

    assert len(groups['black_joker']) == 1
    assert any(b.class_name == 'black_joker' for b in groups['black_joker'])
