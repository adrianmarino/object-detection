def to_list(elements):
    return elements if type(elements) is list else [elements]


def group(elements, resolve_group):
    groups = {}
    for element in elements:
        group = resolve_group(element)
        if group not in groups:
            groups[group] = []
        groups[group].append(element)
    return groups


def by_field_name(name):
    return lambda element: vars(element)[name]


def flatten(main_list):
    flat_list = []
    for sublist in main_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
