import xml.etree.ElementTree as ET
from xml.dom import minidom


def write(root_node, path):
    data = minidom.parseString(ET.tostring(root_node)).toprettyxml(indent="   ")
    with open(path, 'w') as file:
        file.write(data)


def set_property(parent_tag, field, value):
    tag = ET.SubElement(parent_tag, field)
    tag.text = str(value)


def set_int_property(parent_tag, field, value):
    set_property(parent_tag, field, int(round(value)))
