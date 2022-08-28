import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# https://cs.brown.edu/courses/csci1290/labs/lab_raw/index.html
# https://github.com/mushfiqulalam/isp/blob/master/debayer.py


# 坏点矫正
def dead_pixel_correct(input,threshold):

    img = input.astype(np.double)

    m = img.shape[0]
    n = img.shape[1]

    mod_img = img

    # p1 x  p2 x  p3
    # x  x  x  x  x
    # p4 x  p0 x  p6
    # x  x  x  x  x
    # p7 x  p8 x  p9
    
    for i in range(2,m-2):
        for j in range(2,n-2):
            p1, p2, p3 = img[i-2,j-2],img[i-2,j],img[i-2,j+2]
            p4, p0, p6 = img[i,j-2],img[i,j],img[i,j+2]
            p7, p8, p9 = img[i+2,j-2],img[i+2,j],img[i+2,j+2]
            # 1. 坏点检测
            if (abs(p1 - p0) > threshold) and (abs(p2 - p0) > threshold) and (abs(p3 - p0) > threshold) \
                and (abs(p4 - p0) > threshold) and (abs(p6 - p0) > threshold) \
                    and (abs(p7 - p0) > threshold) and (abs(p8 - p0) > threshold) and (abs(p9 - p0) > threshold):
                    # 2. 坏点矫正
                    
                    # 均值方法
                    # p0 = 0.25 * (p2 + p4 + p6 + p8)
                    
                    # 梯度法
                    dv = abs(2 * p0 - p2 - p8) 
                    dh = abs(2 * p0 - p4 - p6)
                    ddr = abs(2 * p0 - p1 - p9)
                    ddl = abs(2 * p0 - p3 - p7)

                    min_d = min(dv,dh,ddr,ddl)
                    if min_d == dv :
                        p0 = (p2+p8+1)/2
                    elif min_d == dh:
                        p0 = (p4+p6+1)/2
                    elif min_d == ddr:
                        p0 = (p1+p9+1)/2
                    else:
                        p0 = (p3+p7+1)/2

                    mod_img[i,j] = p0
    return mod_img

# 黑电平矫正
def black_level_correct(input):
    TODO

# 白平衡
def white_channel_gain(data, channel_gain):
    
    # 计算 Channel gain

    data = data.astype(np.float32)
    channel_gain = np.float32(channel_gain)

    # multiply with the channel gains
    data[::2, ::2]   = data[::2, ::2] * channel_gain[0]
    data[::2, 1::2]  = data[::2, 1::2] * channel_gain[1]
    data[1::2, ::2]  = data[1::2, ::2] * channel_gain[2]
    data[1::2, 1::2] = data[1::2, 1::2] * channel_gain[3]

    return data

# 阴影矫正
def lens_shading_correct(input):
    # 1.矫正系数的标定
    # 2.阴影矫正

# ABF 滤波
def adaptive_Bayer_Filter(data):

# 双边滤波
def bilateral_filter(data,radius,sigmma_color,sigma_space):
    # https://www.cnblogs.com/wangguchangqing/p/6416401.html?tdsourcetag=s_pcqq_aiomsg
    w, h = data.shape[0],data.shape[1]




# CFA插值 - 双线性插值
def demosaicing_bilinear(input):
    img = input.astype(np.double)

    m = img.shape[0]
    n = img.shape[1]

    red_mask = np.tile([[1,0],[0,0]], (int(m/2), int(n/2)))
    green_mask = np.tile([[0,1],[1,0]], (int(m/2), int(n/2)))
    blue_mask = np.tile([[0,0],[0,1]], (int(m/2), int(n/2)))

    r = np.multiply(img, red_mask)
    g = np.multiply(img, green_mask)
    b = np.multiply(img, blue_mask)
    
    filter_g = 0.25 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_g = convolve2d(g, filter_g, 'same')
    g = g + missing_g

    filter1 = 0.25 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    missing_b1 = convolve2d(b, filter1, 'same')

    filter2 = 0.5 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_b2 = convolve2d(b + missing_b1, filter2, 'same')
    b = b + missing_b1 + missing_b2

    filter1 = 0.25 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    missing_r1 = convolve2d(r, filter1, 'same')
    filter2 = 0.5 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_r2 = convolve2d(r + missing_r1, filter2, 'same')
    r = r + missing_r1 + missing_r2

    output = np.stack((r,g,b), axis=2)
    return output

# CFA插值 - Malvar-He-Cutler Linear Image Demosaicking
def demosaicing_malvar(input):

    img = input.astype(np.double)

    w = img.shape[0]
    h = img.shape[1]

    mask_r = np.tile([[1,0],[0,0]], (int(w/2), int(h/2)))
    mask_gr = np.tile([[0,1],[0,0]], (int(w/2), int(h/2)))
    mask_gb = np.tile([[0,0],[1,0]], (int(w/2), int(h/2)))
    mask_b = np.tile([[0,0],[0,1]], (int(w/2), int(h/2)))

    r = np.multiply(img, mask_r)
    g1, g2 = np.multiply(img, mask_gr), np.multiply(img, mask_gr)
    b = np.multiply(img, mask_b)
    

    # G commponent at R location
    filter1 = 0.125 * np.array(
        [
            [0,0,-1,0,0],
            [0,0,2,0,0],
            [-1,2,4,2,-1],
            [0,0,2,0,0],
            [0,0,-1,0,0],
        ])
    filter_tmp = convolve2d(img, filter1, 'same')
    # G component in R location
    gr = np.multiply(mask_r, filter_tmp)
    # G component in B location
    gb = np.multiply(mask_b, filter_tmp)

    filter2 = 0.0625 * np.array(
        [
            [0,0,1,0,0],
            [0,-2,0,-2,0],
            [-2,8,10,8,-2],
            [0,-2,0,-2,0],
            [0,0,1,0,0],
        ])
    filter_tmp = convolve2d(img, filter2, 'same')
    # R component in Gr location
    rgr = np.multiply(mask_gr, filter_tmp)
    # B component in Gr location
    bgr = np.multiply(mask_gr, filter_tmp)


    filter3 = filter2.transpose()
    filter_tmp = convolve2d(img, filter3, 'same')
    # R component in Gb location
    rgb = np.multiply(mask_gb, filter_tmp)
    # B component in Gb location
    bgb = np.multiply(mask_gb, filter_tmp)


    filter4 = 0.0625 * np.array(
        [
            [0,0,-3,0,0],
            [0,4,0,4,0],
            [-3,0,12,0,-3],
            [0,4,0,4,0],
            [0,0,-3,0,0],
        ])

    filter_tmp = convolve2d(img, filter4, 'same')
    # R component in B location
    rb = np.multiply(mask_b, filter_tmp)
    # B component in R location
    br = np.multiply(mask_r, filter_tmp)

    r = rb + rgb + rgr + r
    print("r in b pixel")
    print(rb[:8,:8])
    # print(rgb[:8,:8])
    # print(rgr[:8,:8])
    # print(r[:8,:8])
    b = br + bgb + bgr + b
    g = gr + gb + g1 + g2

    output = np.stack((r,g,b), axis=2)
    return output



# Main pipeline

def pipeline():
    


    black = 0
    saturation = 1023
    multipliers = [2.206157,1.000000,1.210804,1.000000]

    # 读取 json 配置文件
    
    with open("config.json",'r') as f:
        config = json.load(f)

    # step 1: read raw data

    raw_data = Image.open(config['raw_path'])
    # raw = np.fromfile(config['raw_path'], dtype='uint16', sep='')
    # raw = raw.reshape([raw_h, raw_h])
    # raw_data.show()

    raw = np.array(raw_data).astype(np.double)


    raw = raw / 1024

    plt.imshow(raw, cmap='gray')
    plt.show()


    # step 2: demosaicing

    lin_rgb = debayering(raw)
    plt.imshow(lin_rgb)
    plt.show()
