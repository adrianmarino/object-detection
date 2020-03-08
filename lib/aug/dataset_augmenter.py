import abc


class IDataAugmenter:
    @abc.abstractmethod
    def augment(self,  samples, aug_strategy, count):
        pass
