import math
import cv2
import numpy as np
from numpy import exp as exp 
from numpy import cos as cos
from numpy import sin as sin
from numpy import tan as tan 
from numpy import sqrt as sqrt
from numpy import arctan2 as arctan2
from matplotlib import pyplot as plt
import os
import datetime

import zernike_functions
from zernike_functions import zernike


class PreProcess:
    def mkdir_out_dir(dir_fn, fn, N, p, z):
        dt_now = datetime.datetime.now()
        time_str = dt_now.strftime('%Y%m%d-%H%M%S')

        holo_size = int(float(N) * float(p) * float(pow(10.0,3)))
        info_name = 'z'+str(int(z*pow(10,3)))+'mm'+'_size'+str(holo_size)+'mm'

        out_path = './output_imgs/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        out_path = out_path + dir_fn + '/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        out_path = out_path + 'prop_out_' + time_str + '_' + fn + '_' + info_name + '/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        return out_path


class ImageProcess:
    def add_zero_padding(img):
        M = img.shape[0]
        pad_img = np.pad(img, ((M//2,M//2), (M//2,M//2)), 'constant')
        return pad_img

    def remove_zero_padding(img):
        N = img.shape[0]
        M = N // 2
        start_num = M//2
        end_num = N - M//2
        return img[start_num:end_num, start_num:end_num]

    def show_imgs(imgs):
        for i in range(len(imgs)):
            img = imgs[i]
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.gray()
        plt.show()

    def save_imgs(out_dir, imgs):
        for i in range(len(imgs)):
            img = imgs[i]
            cv2.imwrite(out_dir + str(i) + '.png', img)

    def normalize(img):
        img = img / img.max() * 255
        return img

    def normalize_amp_one(img):
        img_max_val = CGH.amplitude(img).max()
        norm_img = img / img_max_val
        return norm_img 

    

class CGH:
    def response(N, l_ambda, z, p):
        phase = np.zeros((N,N), dtype=np.float)
        h = np.zeros((N,N), dtype=np.complex)

        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_y = array_x.T

        # not use np.sqrt
        # phase = np.sqrt(np.power(array_x*p, 2) + np.power(array_y*p, 2)) * np.pi / (l_ambda * z)
        phase = (np.power(array_x*p, 2) + np.power(array_y*p, 2)) * np.pi / (l_ambda * z)
        h = np.exp(1j*phase)

        return h


    def fresnel_fft(u1, N, l_ambda, z, p):
        u1_shift = np.fft.fftshift(u1)
        # f_u1 = np.fft.fft2(u1_shift, (N,N))
        f_u1 = np.fft.fftshift(np.fft.fft2(u1_shift, (N,N)))

        # h: inpulse response
        h = np.zeros((N, N), dtype=np.complex)
        h = CGH.response(N, l_ambda, z, p)

        h_shift = np.fft.fftshift(h)
        # f_h = np.fft.fft2(h_shift, (N,N))
        f_h = np.fft.fftshift(np.fft.fft2(h_shift, (N,N)))

        mul_fft = f_u1 * f_h

        # ifft_mul = np.fft.ifft2(mul_fft)
        # fresnel_img = np.fft.fftshift(ifft_mul)
        fresnel_img = np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift(mul_fft) ) )

        return fresnel_img


    def propagation(u1, N, l_ambda, z, p):
        fresnel_img = CGH.fresnel_fft(u1, N, l_ambda, z, p)
        prop_img = np.exp(1j * (2 * np.pi) * z / l_ambda ) / (1j * l_ambda * z) * fresnel_img
        return prop_img


    def fraunhofer_diffraction(org_img):
        f = np.fft.fft2(org_img)
        fshift = np.fft.fftshift(f)
        return fshift


    def shift_scale_propagation(u1, N, l_ambda, z, p, shift_x=0.0, shift_y=0.0, scale=1.0):
        k = 2*np.pi/l_ambda
        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_x = array_x*p
        array_y = array_x.T

        Cz = np.zeros((N, N), dtype=np.complex)
        Cz = np.exp(1j*k*z) / (1j*l_ambda*z) \
                * np.exp( 1j*np.pi / (l_ambda*z) * ((1-scale)*(np.power(array_x,2)) + 2*shift_x*array_x + shift_x**2) ) \
                * np.exp( 1j*np.pi / (l_ambda*z) * ((1-scale)*(np.power(array_y,2)) + 2*shift_y*array_y + shift_y**2) )

        exp_phi_u = np.zeros((N, N), dtype=np.complex)
        exp_phi_u = np.exp( 1j*np.pi * ((scale**2-scale)*(np.power(array_x,2)) - 2*scale*shift_x*array_x) / (l_ambda*z) ) \
                    * np.exp( 1j*np.pi * ((scale**2-scale)*(np.power(array_y,2)) - 2*scale*shift_y*array_y) / (l_ambda*z) )

        exp_phi_h = np.zeros((N, N), dtype=np.complex)
        exp_phi_h = np.exp( 1j*np.pi * (scale*(np.power(array_x,2)) + scale*(np.power(array_y,2))) / (l_ambda*z) )

        ### 2020.05.17 change order of {fft, fftshift}
        u1 = u1 * exp_phi_u
        u1_shift = np.fft.fftshift(u1)
        f_u1 = np.fft.fft2(u1_shift, (N,N))

        h_shift = np.fft.fftshift(exp_phi_h)
        f_h = np.fft.fft2(h_shift, (N,N))

        mul_fft = f_u1 * f_h

        ifft_mul = np.fft.ifft2(mul_fft)
        ifft_mul_shift = np.fft.fftshift(ifft_mul)

        u2 = Cz * ifft_mul_shift

        return u2


    # https://github.com/thu12zwh/band-extended-angular-spectrum-method
    def angular_spectrum(u1, N, l_ambda, z, p):
        u1_shift = np.fft.fftshift(u1)
        f_u1 = np.fft.fftshift(np.fft.fft2(u1_shift, (N,N)))
        # f_u1 = np.fft.fft2(u1_shift, (N,N))

        # Angular Spectrum
        k = 2.0*np.pi/l_ambda
        if N%2==0:
            array_fx = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_fx = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_fx = array_fx / array_fx.max() / 2.0 / p
        array_fy = array_fx.T
        fx2_fy2 = np.power(array_fx, 2) + np.power(array_fy, 2)
        h = np.exp( 1j*k*z*sqrt(1-fx2_fy2*(l_ambda**2)) )

        mul_fft = f_u1 * h
        u2 = np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift(mul_fft) ) )
        # u2 = np.fft.ifftshift( np.fft.ifft2( mul_fft ) )

        return u2


    def band_limited_angular_spectrum(u1, N, l_ambda, z, p):
        u1_shift = np.fft.fftshift(u1)
        f_u1 = np.fft.fftshift(np.fft.fft2(u1_shift, (N,N)))
        # f_u1 = np.fft.fft2(u1_shift, (N,N))

        # Angular Spectrum
        k = 2.0*np.pi/l_ambda
        if N%2==0:
            array_fx = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
        else:
            array_fx = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
        array_fx = array_fx / array_fx.max() / 2.0 / p
        array_fy = array_fx.T
        fx2_fy2 = np.power(array_fx, 2) + np.power(array_fy, 2)
        h = np.exp( 1j*k*z*sqrt(1-fx2_fy2*(l_ambda**2)) )

        fc = N*p/l_ambda/z/2
        h = h * (~(np.abs(np.sqrt(fx2_fy2)) > fc)).astype(np.int)

        mul_fft = f_u1 * h
        u2 = np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift(mul_fft) ) )
        # u2 = np.fft.ifftshift( np.fft.ifft2( mul_fft ) )

        return u2


    def amplitude(img):
        amp_img = sqrt( img.real * img.real + img.imag * img.imag )
        return amp_img


    def phase(img):
        height = img.shape[0]
        width = img.shape[1]
        phase_cgh = np.zeros((height,width), dtype=np.float)
        re = img.real
        im = img.imag
        phase_cgh = arctan2(im, re)
        return phase_cgh


    def anti_phase(phase_img):
        anti_phase_img = phase_img * -1.0
        return anti_phase_img


    def intensity(img):
        intensity = img.real * img.real + img.imag * img.imag
        return intensity


    def exp_cgh(N, p, z, k, x_j, y_j):
        r_map = np.zeros((N,N))
        exp_cgh = np.empty((N,N), dtype='complex128')
        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        array_y = array_x.T
        r_map = sqrt( np.power((array_x-x_j)*p, 2) + np.power((array_y-y_j)*p, 2) + z**2 )
        exp_cgh = 1/r_map * exp(1j*k*r_map)
        return exp_cgh


    def normalize_exp_cgh(exp_cgh):
        max_amp_abs = np.abs(CGH.amplitude(exp_cgh)).max()
        norm_exp_cgh = exp_cgh / max_amp_abs
        return norm_exp_cgh


    def exp_cgh_zone_limit(N, p, z, k, x_j, y_j, limit_len, exp_cgh_img):
        for y in range(N):
            for x in range(N):
                if sqrt( pow((x-x_j)*p,2) + pow((y-y_j)*p,2) ) > limit_len:
                    exp_cgh_img[y][x] = 0
        return exp_cgh_img


    def amp_cgh(N, p, z, k, x_j, y_j):
        r_map = np.zeros((N,N))
        amp_cgh = np.zeros((N,N))
        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        array_y = array_x.T
        r_map = sqrt( np.power((array_x-x_j)*p, 2) + np.power((array_y-y_j)*p, 2) + z**2 )
        amp_cgh = 1/r_map * cos(k*r_map)
        return amp_cgh


    def amp_cgh_zone_limit(N, p, z, k, x_j, y_j, limit_len, amp_cgh_img):
        for y in range(N):
            for x in range(N):
                if sqrt( pow((x-x_j)*p,2) + pow((y-y_j)*p,2) ) > limit_len:
                    amp_cgh_img[y][x] = 255
        return amp_cgh_img


    def phase_cgh(N, p, z, k, x_j, y_j):
        r_map = np.zeros((N,N))
        phase_cgh = np.zeros((N,N))
        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        array_y = array_x.T
        r_map = sqrt( np.power((array_x-x_j)*p, 2) + np.power((array_y-y_j)*p, 2) + z**2 )
        re = 1/r_map * cos(k*r_map)
        im = 1/r_map * sin(k*r_map)
        phase_cgh = arctan2(im, re)
        phase_cgh = ( phase_cgh / np.pi / 2.0 + 0.5 ) * 255
        return phase_cgh 


    def phase_cgh_pi(N, p, z, k, x_j, y_j):
        r_map = np.zeros((N,N))
        phase_cgh = np.zeros((N,N))
        if N%2==0:
            array_x = np.array([i % N - (N//2-0.5) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        else:
            array_x = np.array([i % N - (N//2) for i in range(N * N)]).reshape(N, N)
            x_j = x_j % N - (N//2-0.5) 
            y_j = y_j % N - (N//2-0.5) 
        array_y = array_x.T
        r_map = sqrt( np.power((array_x-x_j)*p, 2) + np.power((array_y-y_j)*p, 2) + z**2 )
        re = 1/r_map * cos(k*r_map)
        im = 1/r_map * sin(k*r_map)
        phase_cgh = arctan2(im, re)
        return phase_cgh 


    def phase_norm(phase_pi):
        height = phase_pi.shape[0]
        width  = phase_pi.shape[1]
        phase_norm = np.zeros((height,width))
        phase_norm = (phase_pi + np.pi) / (2.0*np.pi) * 255
        return phase_norm


    def phase_from_img(phase_img):
        height = phase_img.shape[0]
        width  = phase_img.shape[1]
        phase_pi = np.zeros((height,width))
        phase_pi = (phase_img / 255.0 * 2.0 - 1.0) * np.pi
        return phase_pi


    def amp_abs(amp_cgh_img):
        new_amp_cgh = amp_cgh_img + abs(amp_cgh_img.min())
        new_amp_cgh = new_amp_cgh / new_amp_cgh.max() * 255
        return new_amp_cgh


    def pupil_func(N, p, pupil_r_m):
        p_xy = np.zeros((N, N))
        screen_size = N * p
        half_size = screen_size / 2.0
        x = np.linspace(-1 * half_size, half_size, N)
        y = np.linspace(-1 * half_size, half_size, N)
        [X,Y] = np.meshgrid(x,y)
        r_map = sqrt(X**2+Y**2)
        p_xy = np.where(r_map>pupil_r_m, 0.0, 1.0)
        return p_xy


    def big_pupil_func(N):
        p_xy = np.full((N,N), 1.0)
        return p_xy


    def norm_wave_aberration(N, nm):
        W_xy = np.zeros((N, N))
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        [X,Y] = np.meshgrid(x,y)
        r_map = sqrt(X**2+Y**2)
        theta_map = np.arctan2(Y,X)
        W_xy = zernike.z_func(nm[0], nm[1], r_map, theta_map, N)

        r_map = np.where(r_map>1.0, 0.0, 1.0)
        W_xy = W_xy * r_map

        return W_xy


    def resize_and_add_pad(W_xy, N, p, pupil_r_m):
        pupil_pix = int(pupil_r_m / p * 2)
        resized_W_xy = cv2.resize(W_xy, dsize=(pupil_pix, pupil_pix))
        add_len01 = int((N - pupil_pix) / 2)
        add_len02 = N - pupil_pix - add_len01
        pad_img = np.pad(resized_W_xy, ((add_len01,add_len02), (add_len01,add_len02)), 'constant')

        return pad_img


    def resize_and_add_pad_rect(W_xy, N, p, pupil_r_m):
        pupil_pix = int(pupil_r_m / p * 2)
        resized_W_xy = cv2.resize(W_xy, dsize=(pupil_pix, pupil_pix))
        add_len01 = int((N - pupil_pix) / 2)
        add_len02 = N - pupil_pix - add_len01
        pad_img = np.pad(resized_W_xy, ((add_len01,add_len02), (add_len01,add_len02)), 'constant')

        return pad_img


    def general_pupil(p_xy, W_xy, l_ambda):
        k = 2*np.pi/l_ambda
        # W_xy = W_xy * pow(10,-9)
        W_xy = W_xy * pow(10,-6)
        # W_xy = W_xy * pow(10,-3)
        pupil_img = p_xy * np.exp(1j*k*W_xy)
        return pupil_img


    def lens_txy(u_minus, P_xy, l_ambda, lens_f, p, N):
        lens = CGH.response(N, l_ambda, -1.0*lens_f, p)
        u_plus = u_minus * lens * P_xy
        u_plus += 0.0
        return u_plus


    def not_aliasing(z, N, p, l_ambda):
        if z >= N * (p**2) / l_ambda:
            return True
        else:
            return False


    def not_aliasing_z(N, p, l_ambda):
        z = N * (p**2) / l_ambda
        return z


    def not_aliasing_area(l_ambda, z, p):
        pix_cnt = l_ambda * z / (2*p) / p * 2
        return pix_cnt


    def check_aliasing(lens_f, N, l_ambda, p, z):
        not_aliasing_bool = CGH.not_aliasing(lens_f, N, p, l_ambda)
        not_aliasing_z_d = CGH.not_aliasing_z(N, p, l_ambda)
        not_aliasing_pix_size = CGH.not_aliasing_area(l_ambda, lens_f, p)

        print(not_aliasing_bool)
        print(not_aliasing_z_d)
        print(not_aliasing_pix_size)


    def DPAC_plane(input_plane):
        N = input_plane.shape[0]
        slm_amp = CGH.amplitude(input_plane)
        slm_phase = CGH.phase(input_plane)

        new_slm_amp = slm_amp + abs(slm_amp.min())
        new_slm_amp = new_slm_amp / new_slm_amp.max() * 1.0

        h1_slm = np.exp( 1j * ( slm_phase + np.arccos(new_slm_amp/2.0) ) )
        h2_slm = np.exp( 1j * ( slm_phase - np.arccos(new_slm_amp/2.0) ) )

        dpac_plane01 = np.zeros((N,N),dtype=bool)
        dpac_plane01[::2,::2] = True
        dpac_plane01[1::2,1::2] = True
        dpac_plane02 = ~(dpac_plane01)
        dpac_plane01 = np.where(dpac_plane01==True, 1, 0)
        dpac_plane02 = np.where(dpac_plane02==True, 1, 0)

        dpac_plane = h1_slm * dpac_plane01
        dpac_plane += h2_slm * dpac_plane02

        return dpac_plane