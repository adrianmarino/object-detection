class Image:
    def __init__(self, path, width, height, depth):
        self.path = path
        self.width = width
        self.height = height
        self.depth = depth

    def __eq__(self, other):
        return self.path == other.path and \
               self.width == other.width and \
               self.height == other.height and \
               self.depth == other.depth
