import os
import re
import xml.etree.ElementTree as et


def read_xml_file(file_path):
    data = open(file_path, 'r', errors='ignore').read()

    parser = et.XMLParser(encoding="utf-8")
    for ue in ['uFRM', 'uSEP', 'uBS', 'uEQ', 'uNOT', 'uGT', 'uLT', 'uALL', 'uAND', 'uOR', 'uONE', 'uB', 'cx7F', 'uI',
               'uBI', 'uIU', 'uU', 'uBU', 'uBIU', 'uPG', '']:
        parser.entity[ue] = ue

    root = et.fromstring(data[data.find('<'):], parser=parser)
    tree = et.ElementTree(root)
    code_elements = [*tree.findall('.//DAT[@name="UTEXT"]'),  # libprc, libinc
                     *tree.findall('.//DAT[@name="UOCC_SCRIPT"]'),  # ent, cpt, aps
                     *tree.findall('.//DAT[@name="USCRIPT"]')]  # ent, cpt, aps
    del root
    del tree
    codes_text = '\n\n'.join([elem.text for elem in code_elements])
    return codes_text


def read_xml_directories(tests_path, code_directories):
    all_files_contents = []
    for cd in code_directories:
        for root, dirs, files in os.walk(os.path.join(tests_path, cd)):
            for name in files:
                text = read_xml_file(os.path.join(root, name)).lower()
                text = re.sub(' +', ' ', text)
                text = re.sub('( *[\r\n])+', '\r\n', text)
                all_files_contents.append(text)

    return all_files_contents
