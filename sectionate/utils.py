import json
import os

from sectionate import Section, GriddedSection

def load_sections_from_catalog(filename):
    """
    Load all sections from a specified catalog JSON file.

    Parameters
    ----------
    filename : str
        The name of the JSON file in the catalog directory.

    Returns
    -------
    dict
        A dictionary mapping section names to Section objects.

    Examples
    --------
    >>> sectionate.utils.load_sections_from_catalog("Atlantic_transport_arrays.json")
    """
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
    """
    Get a list of all JSON catalog files in the catalog directory.

    Returns
    -------
    list of str
        List of JSON filenames found in the catalog directory.
        Returns an empty list if the directory does not exist.

    Example
    -------
    >>> sectionate.utils.get_section_catalog()
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_dir = os.path.join(base_dir, 'catalog')
    if not os.path.isdir(catalog_dir):
        print("catalog directory does not exist.")
        return []
    json_files = [f for f in os.listdir(catalog_dir) if f.endswith('.json')]
    return json_files

def get_all_section_names():
    """
    Get a list of all section names available in all catalog JSON files.

    Returns
    -------
    list of str
        List of all unique section names found in the catalog.

    Example
    -------
    >>> sectionate.utils.get_all_section_names()
    """
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
    """
    Load a section by name from the section catalog.

    Searches through all JSON files in the section catalog directory for the specified
    section name. If found, returns a Section object initialized with the section's
    longitude and latitude values.

    Parameters
    ----------
    section_name : str
        The name of the section to load.

    Returns
    -------
    Section or None
        A Section object if the section is found, otherwise None.

    Example
    -------
    >>> section = sectionate.load_section("OSNAP West")
    """
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

def save_gridded_section(filepath, gridded_section):
    """
    Save a GriddedSection object to a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file to write.
    gridded_section : GriddedSection
        The GriddedSection object to save.

    Returns
    -------
    None
    """
    data = {
        "name": gridded_section.name,
        "lons_c": [float(x) for x in gridded_section.lons_c],
        "lats_c": [float(x) for x in gridded_section.lats_c],
        "i_c": [int(x) for x in gridded_section.i_c],
        "j_c": [int(x) for x in gridded_section.j_c]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_gridded_section(filepath, grid):
    """
    Load a GriddedSection from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file containing the gridded section data.
    grid : xgcm.Grid
        The xgcm Grid object to associate with the section.

    Returns
    -------
    GriddedSection
        The loaded GriddedSection object.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    name = data["name"]
    coords = (data["lons_c"], data["lats_c"])
    section = Section(name,coords)
    return GriddedSection(
        section,
        grid,
        i_c=data["i_c"],
        j_c=data["j_c"],
    )