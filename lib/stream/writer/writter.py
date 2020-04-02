import abc


class Writer(abc.ABC):

    @abc.abstractmethod
    def write(self, frame):
        pass

    @abc.abstractmethod
    def close(self):
        pass
