from xml.dom import minidom
import xml.etree.ElementTree as ET

def write(rootNode, path):
    data = minidom.parseString(ET.tostring(rootNode)).toprettyxml(indent="   ")
    with open(path, 'w') as file:
        file.write(data)

def property(parentTag, field, value):
    tag = ET.SubElement(parentTag, field)
    tag.text = str(value)
