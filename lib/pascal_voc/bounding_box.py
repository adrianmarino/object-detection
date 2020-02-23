class BoundingBox:
    def __init__(self, class_name, xmin, ymin, xmax, ymax):
        self.class_name = class_name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __eq__(self, other):
        return self.class_name == other.class_name and \
               self.xmin == other.xmin and \
               self.ymin == other.ymin and \
               self.xmax == other.xmax and \
               self.ymax == other.ymax
