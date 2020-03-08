from imgaug import augmenters as iaa


class DataAugSequenceFactory:

    @staticmethod
    def sequence_a():
        return iaa.Sequential([
            iaa.GammaContrast(1.5),
            iaa.Affine(rotate=10)
        ])
