import imgaug as ia
import imgaug.augmenters as iaa


class DataAugSequenceFactory:

    @staticmethod
    def sequence_a():
        return iaa.Sequential([
            iaa.GammaContrast(1.5),
            iaa.Affine(rotate=10)
        ])

    @staticmethod
    def sequence_b():
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        return iaa.Sequential(
            [
                iaa.Resize({'height': 'keep-aspect-ratio', 'width': 0.065}),

                iaa.Sequential(
                    [
                        #
                        # Apply the following augmenters to most images.
                        #

                        # crop some of the images by 0-10% of their height/width
                        sometimes(
                            iaa.Sequential([
                                iaa.Crop(percent=(0, 0.1)),
                                iaa.RemoveCBAsByOutOfImageFraction(0.6),
                                iaa.ClipCBAsToImagePlanes()
                            ])
                        ),

                        # Apply affine transformations to some of the images
                        # - scale to 80-120% of image height/width (each axis independently)
                        # - translate by -20 to +20 relative to height/width (per axis)
                        # - rotate by -45 to +45 degrees
                        # - shear by -16 to +16 degrees
                        # - order: use nearest neighbour or bilinear interpolation (fast)
                        # - mode: use any available mode to fill newly created pixels
                        #         see API or scikit-image for which modes are available
                        # - cval: if the mode is constant, then use a random brightness
                        #         for the newly created pixels (e.g. sometimes black,
                        #         sometimes white)
                        sometimes(
                            iaa.Sequential([
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-70, 70),
                                    shear=(-16, 16),
                                    order=[0, 1],
                                    cval=(0, 255),
                                    mode="edge"
                                ),
                                iaa.RemoveCBAsByOutOfImageFraction(0.6),
                                iaa.ClipCBAsToImagePlanes()
                            ])
                        ),

                        #
                        # Execute 0 to 5 of the following (less important) augmenters per
                        # image. Don't execute all of them, as that would often be way too
                        # strong.
                        #
                        iaa.SomeOf((0, 5),
                                   [
                                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                                       # them to black) or drop them on an image with 2-5% percent
                                       # of the original size, leading to large dropped
                                       # rectangles.
                                       iaa.OneOf([
                                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                           iaa.CoarseDropout(
                                               (0.03, 0.05), size_percent=(0.01, 0.02),
                                               per_channel=0.2
                                           ),
                                       ]),

                                       # Invert each image's channel with 5% probability.
                                       # This sets each pixel value v to 255-v.
                                       # iaa.Invert(0.05, per_channel=True),  # invert color channels

                                       # Add a value of -10 to 10 to each pixel.
                                       iaa.Add((-5, 5), per_channel=0.5),

                                       # Change brightness of images (50-150% of original value).
                                       iaa.Multiply((0.5, 2.0), per_channel=0.5),

                                       # Improve or worsen the contrast of images.
                                       iaa.LinearContrast((0.1, 1.0), per_channel=0.5),

                                       # Convert each image to grayscale and then overlay the
                                       # result with the original with random alpha. I.e. remove
                                       # colors with varying strengths.
                                       iaa.Grayscale(alpha=(0.2, 1.0)),
                                   ],
                                   # do all of the above augmentations in random order
                                   random_order=True
                                   )
                    ],
                    # do all of the above augmentations in random order
                    random_order=True
                )
            ]
        )
