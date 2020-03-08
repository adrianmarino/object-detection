import abc


class SampleAugStrategy:
    @abc.abstractmethod
    def augment(self, image, bboxes):
        pass
