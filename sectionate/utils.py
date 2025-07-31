import json
import os

from sectionate import Section

def load_sections_from_catalog(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.join(base_dir, 'catalog', filename)
    with open(catalog_path, 'r') as f:
        content = json.load(f)
        section_dict = {
            section_name: Section(
                section_name,
                (values["lon"], values["lat"])
            )
            for (section_name, values) in content.items()
        }
        return section_dict

def get_section_catalog():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_dir = os.path.join(base_dir, 'catalog')
    if not os.path.isdir(catalog_dir):
        print("catalog directory does not exist.")
        return
    json_files = [f for f in os.listdir(catalog_dir) if f.endswith('.json')]
    return json_files

def get_all_section_names():
    section_names = set()
    json_files = get_section_catalog()
    if not json_files:
        return []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_dir = os.path.join(base_dir, 'catalog')
    for filename in json_files:
        catalog_path = os.path.join(catalog_dir, filename)
        with open(catalog_path, 'r') as f:
            content = json.load(f)
            section_names.update(content.keys())
    return list(section_names)

def load_section(section_name):
    json_files = get_section_catalog()
    if not json_files:
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_dir = os.path.join(base_dir, 'catalog')
    for filename in json_files:
        catalog_path = os.path.join(catalog_dir, filename)
        with open(catalog_path, 'r') as f:
            content = json.load(f)
            if section_name in content:
                values = content[section_name]
                return Section(section_name, (values["lon"], values["lat"]))
    return None