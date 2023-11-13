#color_moments.py>

import numpy as np
from PIL import Image
import math
import util

def get_ColorMoments(image) :
    
    image_h = 100
    image_w = 300
    
    image = image.resize((image_w,image_h))
    
    block_h = int(image_h / 10)
    block_w = int(image_w / 10)
    
    col_moments = np.zeros((3,3,10,10))#[[[[[0] for _ in range(10)] for _ in range(10)] for _ in range(3)] for _ in range(3)]

    for i in range(0,image_h,block_h) :
        for j in range(0,image_w,block_w) :
            
            #i iterate in [0,10,20,...,90] and j in [0,30,60,...,270]
            #[i,j] and [i+9,j+29] are the limits of the block in question
            #In next step we go through each pixel in that block,store it in block[3*10*30] and get the sum of each rgb value
            #which will help us find mean, sd and skew 
            
            block = np.zeros((3,block_h,block_w))
            rgb_sum = [0,0,0]
            for k in range(i,i+10,1) :
                for l in range(j,j+30,1) :
                    r,g,b = image.getpixel((l,k))
                    
                    block[0][k-i][l-j] = r
                    block[1][k-i][l-j] = g
                    block[2][k-i][l-j] = b
        
                    rgb_sum[0] += r
                    rgb_sum[2] += b
                    rgb_sum[1] += g
            
            #calculate mean
            rgb_mean = [rgb_sum[0]/(block_h*block_w),rgb_sum[1]/(block_h*block_w),rgb_sum[2]/(block_h*block_w)]
            
            #rgb_2s : stores sum of squares 
            #rgb_3s : stores cube of squares
            rgb_2s,rgb_3s = [0,0,0],[0,0,0]
            for k in range(0,block_h,1) :
                for l in range(0,block_w,1) :
                    for col in range(3) :
                        rgb_2s[col] += (block[col][k][l]-rgb_mean[col])**2
                        rgb_3s[col] += (block[col][k][l]-rgb_mean[col])**3
            
            #[r,g,b] * [mean,sd,skew] * [10*10]
            for col in range(3) :
                col_moments[col][0][int(i/block_h)][int(j/block_w)] = rgb_mean[col]
                col_moments[col][1][int(i/block_h)][int(j/block_w)] = math.sqrt(rgb_2s[col]/(block_h*block_w))
                col_moments[col][2][int(i/block_h)][int(j/block_w)] = util.cube_root(rgb_3s[col]/(block_h*block_w))

    return col_moments