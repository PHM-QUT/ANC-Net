import numpy as np
import math
from torch.nn import Module
from .DWT_IDWT_Functions import *
import pywt

__all__ = ['DWT_1D', 'IDWT_1D']
class DWT_1D(Module):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    """
    def __init__(self, args, wavename):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        用于序列分解的一维离散小波变换 DWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_1D, self).__init__()
        self.args = args
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            # self.matrix_low = torch.Tensor(matrix_h).cuda()
            # self.matrix_high = torch.Tensor(matrix_g).cuda()
            self.matrix_low = torch.Tensor(matrix_h).to(self.args)
            self.matrix_high = torch.Tensor(matrix_g).to(self.args)
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, input):
        """
        input_low_frequency_component = \mathcal{L} * input
        input_high_frequency_component = \mathcal{H} * input
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """
    def __init__(self, wavename):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        用于序列重构的一维离散小波逆变换 IDWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        生成变换矩阵
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


if __name__ == '__main__':
    from datetime import datetime
    from torch.autograd import gradcheck
    wavelet = pywt.Wavelet('bior1.1')
    h = wavelet.rec_lo
    g = wavelet.rec_hi
    h_ = wavelet.dec_lo
    g_ = wavelet.dec_hi
    h_.reverse()
    g_.reverse()

    """
    image_full_name = '/home/li-qiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    height, width, channel = image.shape
    #image = image.reshape((1,height,width))
    t0 = datetime.now()
    for index in range(100):
        m0 = DWT_2D(band_low = h, band_high = g)
        image_tensor = torch.Tensor(image)
        image_tensor.unsqueeze_(dim = 0)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(1,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(2,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LL, LH, HL, HH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        #test = gradcheck(DWTFunction_2D.apply, input)
        #print(test)
        #print(LL.requires_grad)
        #print(LH.requires_grad)
        #print(HL.requires_grad)
        #print(HH.requires_grad)
        #LL.requires_grad = True
        #input = (LL.double(), LH.double(), HL.double(), HH.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        #test = gradcheck(IDWTFunction_2D.apply, input)
        #print(test)

        m1 = IDWT_2D(band_low = h_, band_high = g_)
        image_re = m1(LL,LH,HL,HH)
    t1 = datetime.now()
    image_re.transpose_(2,3)
    image_re.transpose_(1,3)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[0]
    print(np.max(image_zero), np.min(image_zero))
    print(image_zero[:,8])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[0]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """
    """
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    image_3d = np.concatenate((image, image, image, image, image, image), axis = 2)
    print(image_3d.shape)
    image_tensor = torch.Tensor(image_3d)
    #image_tensor = image_tensor.transpose(dim0 = 2, dim1 = 1)
    #image_tensor = image_tensor.transpose(dim0 = 1, dim1 = 0)
    image_tensor.unsqueeze_(dim = 0)
    image_tensor.unsqueeze_(dim = 0)
    t0 = datetime.now()
    for index in range(10):
        m0 = DWT_3D(wavename = 'haar')
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_low_2 = torch.Tensor(m0.matrix_low_2)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)
        matrix_high_2 = torch.Tensor(m0.matrix_high_2)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #                                matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(DWTFunction_3D.apply, input)
        #print('testing dwt3d -- {}'.format(test))
        #LLL.requires_grad = True
        #input = (LLL.double(), LLH.double(), LHL.double(), LHH.double(),
        #         HLL.double(), HLH.double(), HHL.double(), HHH.double(),
        #         matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #         matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(IDWTFunction_3D.apply, input)
        #print('testing idwt3d -- {}'.format(test))

        m1 = IDWT_3D(wavename = 'haar')
        image_re = m1(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH)
    t1 = datetime.now()
    image_re.squeeze_(dim = 0)
    image_re.squeeze_(dim = 0)
    #image_re.transpose_(0,1)
    #image_re.transpose_(1,2)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[:,:,0:3]
    print(np.max(image_zero), np.min(image_zero))
    #print(image_zero[:,8,0])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[:,:,0:3]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """

    """
    import matplotlib.pyplot as plt
    import numpy as np
    vector_np = np.array(list(range(1280)))#.reshape((128,1))

    print(vector_np.shape)
    t0 = datetime.now()
    for index in range(100):
        vector = torch.Tensor(vector_np)
        vector.unsqueeze_(dim = 0)
        vector.unsqueeze_(dim = 0)
        m0 = DWT_1D(band_low = h, band_high = g)
        L, H = m0(vector)

        #matrix_low = torch.Tensor(m0.matrix_low)
        #matrix_high = torch.Tensor(m0.matrix_high)
        #vector.requires_grad = True
        #input = (vector.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(DWTFunction_1D.apply, input)
        #print('testing 1D-DWT: {}'.format(test))
        #print(L.requires_grad)
        #print(H.requires_grad)
        #L.requires_grad = True
        #H.requires_grad = True
        #input = (L.double(), H.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(IDWTFunction_1D.apply, input)
        #print('testing 1D-IDWT: {}'.format(test))

        m1 = IDWT_1D(band_low = h_, band_high = g_)
        vector_re = m1(L, H)
    t1 = datetime.now()
    vector_re_np = vector_re.detach().numpy()
    print('image_re shape: {}'.format(vector_re_np.shape))

    vector_zero = vector_np - vector_re_np.reshape(vector_np.shape)
    print(np.max(vector_zero), np.min(vector_zero))
    print(vector_zero[:8])
    print('taking {} secondes'.format(t1 - t0))
    """
