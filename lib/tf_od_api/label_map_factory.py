class LabelMapFactory:
    def create(self, classes):
        label_map = ''
        index = 1
        for a_class in classes:
            label_map += self.__create_item(index, a_class)
            index += 1
        return label_map

    @staticmethod
    def __create_item(index, class_name):
        return 'item {\n\tid: ' + str(index) + '\n\tname: \'' + class_name + '\'\n}\n'
