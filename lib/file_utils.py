def read(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def write(file_path, data):
    with open(file_path, 'wt') as file:
        file.write(data)
