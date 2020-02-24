from lib import file_utils
from lib.config import Config

cfg = Config('../../config.yml')
LABEL_MAP_PATH = 'label_map.pbtxt'


def test_create_label_map():
    # Prepare
    classes = cfg.property('classes')
    factory = LabelMapFactory()
    expected_label_map = file_utils.read(LABEL_MAP_PATH)

    # Perform
    label_map = factory.create(classes)
    # file_utils.write(LABEL_MAP_PATH, label_map)

    # Asserts
    assert label_map == expected_label_map
