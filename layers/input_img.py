class InputImage:
    def __init__(self, sample_img):
        self.__n_channel = len(sample_img)
        self.__dim = len(sample_img[0])

    def get_n_channel(self):
        return self.__n_channel

    def get_n_units(self):
        return self.__dim * self.__dim * self.__n_channel

    def get_dim(self):
        return self.__dim
