import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import sqrt as sqrt
from numpy import arctan2 as arctan2
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.ticker import LinearLocator as LinearLocator
from matplotlib.ticker import FormatStrFormatter as FormatStrFormatter
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
# from . import tools as tools
import math
import sympy as sp
import cv2

### based on this PDF
### https://www.jstage.jst.go.jp/article/jpnjvissci/35/2/35_38/_pdf


class zernike:
    # Znm = Rnm * cos()
    # Define Rnm
    def r_func(n, m, r_map, N):
        length = (n-m)//2
        Rnm = np.zeros((N, N))
        R = np.zeros((N, N))
        for k in range(length+1):
            Rnm = ( (-1)**k * math.factorial(n-k) * r_map**(n-2*k) ) \
                        / \
                    ( math.factorial(k) * math.factorial((n+m)/2 - k) * math.factorial((n-m)/2 - k) )
            R += Rnm
        return R


    # based on this site
    # https://meltingrabbit.com/blog/article/2017122401/
    # W(x,y) = Σc*Z(r,theta)
    def z_func(n, m, r_map, theta_map, N):
        if m > 0:
            Z_val = sqrt(2*(n+1)) * zernike.r_func(n, m, r_map, N) * cos(m * theta_map)
        elif m < 0:
            Z_val = sqrt(2*(n+1)) * zernike.r_func(n, -1 * m, r_map, N) * sin(abs(m) * theta_map)
        elif m == 0:
            Z_val = sqrt(2*(n+1)) * zernike.r_func(n, m, r_map, N) / sqrt(2)

        return Z_val


    def norm_zernike(img_data):
        img_data = img_data + abs(img_data.min())
        img_data = (img_data / img_data.max()) * 255
        img_data.astype('uint16')

        return img_data