import tensorflow as tf
import numpy as np

from functions import *


class tf_CGH:
    def angular_h(k, N, l_ambda, z, p):
        if N%2==0:
            array_fx = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_fx = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_fx = array_fx / array_fx.max() / 2.0 / p
        array_fy = array_fx.T
        fx2_fy2 = np.power(array_fx, 2) + np.power(array_fy, 2)
        h = np.exp( 1j*k*z*np.sqrt(1-fx2_fy2*(l_ambda**2)) )

        h = tf.constant(h)
        return h


    def angular_spectrum(u1, k, N, l_ambda, z, p):
        u1_shift = tf.signal.fftshift(u1)
        f_init = tf.signal.fftshift(tf.signal.fft2d(u1_shift))

        angular = tf_CGH.angular_h(k, N, l_ambda, z, p)

        mul_fft = f_init * angular
        u2 = tf.signal.ifftshift( tf.signal.ifft2d( tf.signal.ifftshift(mul_fft) ) )

        return u2


    def band_limited_angular_h(k, N, l_ambda, z, p):
        if N%2==0:
            array_fx = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_fx = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_fx = array_fx / array_fx.max() / 2.0 / p
        array_fy = array_fx.T
        fx2_fy2 = np.power(array_fx, 2) + np.power(array_fy, 2)
        h = np.exp( 1j*k*z*np.sqrt(1-fx2_fy2*(l_ambda**2)) )

        fc = N*p/l_ambda/z/2
        h = h * (~(np.abs(np.sqrt(fx2_fy2)) > fc)).astype(np.int)

        h = tf.constant(h)
        return h


    def band_limited_angular_spectrum(u1, k, N, l_ambda, z, p):
        u1_shift = tf.signal.fftshift(u1)
        f_init = tf.signal.fftshift(tf.signal.fft2d(u1_shift))

        angular = tf_CGH.band_limited_angular_h(k, N, l_ambda, z, p)

        mul_fft = f_init * angular
        u2 = tf.signal.ifftshift( tf.signal.ifft2d( tf.signal.ifftshift(mul_fft) ) )

        return u2


    def response_h(N, l_ambda, z, p):
        phase = np.zeros((N,N), dtype=np.float)
        h = np.zeros((N,N), dtype=np.complex)

        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_y = array_x.T

        phase = (np.power(array_x*p, 2) + np.power(array_y*p, 2)) * np.pi / (l_ambda * z)
        h = np.exp(1j*phase)
        h = tf.constant(h)
        
        return h


    def add_zero_padding(img):
        M = img.shape[0]
        pad_img = tf.pad(img, ((M//2,M//2), (M//2,M//2)), 'constant')
        return pad_img


    def remove_zero_padding(img):
        N = img.shape[0]
        M = N // 2
        start_num = M//2
        end_num = N - M//2
        return img[start_num:end_num, start_num:end_num]


    def lens_txy(u_minus, P_xy, l_ambda, lens_f, p, N):
        lens = tf_CGH.response_h(N, l_ambda, -1.0*lens_f, p)
        u_plus = u_minus * lens * P_xy
        u_plus += 0.0
        return u_plus

    
    def normalize_amp_one(img):
        img_max_val = CGH.amplitude(img.numpy()).max()
        norm_img = img / img_max_val
        return norm_img 

    
    def fresnel_fft(u1, N, l_ambda, z, p):
        u1_shift = tf.signal.fftshift(u1)
        f_u1 = tf.signal.fftshift(tf.signal.fft2d(u1_shift, (N,N)))

        h = tf_CGH.response_h(N, l_ambda, z, p)

        h_shift = tf.signal.fftshift(h)
        f_h = tf.signal.fftshift(tf.signal.fft2d(h_shift, (N,N)))

        mul_fft = f_u1 * f_h

        fresnel_img = tf.signal.ifftshift( tf.signal.ifft2d( tf.signal.ifftshift(mul_fft) ) )

        return fresnel_img
        

    def propagation(u1, N, l_ambda, z, p):
        fresnel_img = tf_CGH.fresnel_fft(u1, N, l_ambda, z, p)
        exp_val = tf.constant( np.exp(1j * (2 * np.pi) * z / l_ambda ) / (1j * l_ambda * z) )
        prop_img = exp_val * fresnel_img
        return prop_img