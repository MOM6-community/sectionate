def test_load_section():
    from sectionate.utils import get_all_section_names, load_section
    section_names = get_all_section_names()
    load_section(section_names[0])