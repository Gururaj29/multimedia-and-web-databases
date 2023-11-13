#HOG.py>

from PIL import Image
import math
import numpy as np
# import sys 
# sys.path.append('..')
# from Code import util
import util

def calculate_hog_weighted(hog) :
    bin_ = np.zeros(9)
    for i in range(len(hog[0])) :
        for j in range(len(hog[0][0])) :

            #distribute magnitude in two nearest angle bins
            #i.e. angle 13 goes to bin 0 and 1 in ratio 27:13

            angle = (hog[1][i][j] if (hog[1][i][j] >= 0) else (360+hog[1][i][j]))
            ratio = (angle - (int(angle/40)*40)) / abs(angle - ((int(angle/40) + 1))*40)
            bin_[int(angle/40)] += (hog[0][i][j])/(ratio + 1)
            bin_[(int(angle/40) + 1)%9] += (hog[0][i][j]*ratio)/(ratio + 1)
            
    return bin_



def get_HOG(image) :

    image_h = 100
    image_w = 300
    
    image = image.resize((image_w,image_h))
    image = image.convert('L')

    
    hog_vector = [[[0] for _ in range(10)] for _ in range(10)]
    
    #step1 get blocks of 30*10
    #step2 get 30*10*2 where one 30*10 block shows the magnitude and the other shows the angles 
    #step3 Create Histogram based on angles values
    block_height = int(image_h / 10)
    block_width = int(image_w / 10)

    #mag = np.zeros((100,300))
    #ang = np.zeros((100,300))
    #bin_mat = [[[0] for _ in range(10)] for _ in range(10)]

    for i in range(0,image_h,block_height) :
        for j in range(0,image_w,block_width) :
            
            #process block in i,j to i+10,j+29
            #apply horizontal and vertical filter and find gx and gy for each pixel which will determine 
            #direction and magnitude
            
            hog = np.zeros((2,block_height,block_width)) # 0:magnitude 1:angles
            
            for k in range(i,i+block_height,1) :
                for l in range(j,j+block_width,1) : 
                    gx,gy = 0,0

                    #horizontal - apply effect of filter [-1,0,1]
                    if(util.chk(k,l-1,image_h,image_w)) : 
                        gx += (-image.getpixel((l-1,k)))#(-block[k][l-1])
                    if(util.chk(k,l+1,image_h,image_w)) :
                        gx += image.getpixel((l+1,k))#(block[k][l+1])
                    
                    #vertical - apply effect of filter [-1,0,1].T
                    if(util.chk(k-1,l,image_h,image_w)) :
                        gy += (-image.getpixel((l,k-1)))#(-block[k-1][l])
                    if(util.chk(k+1,l,image_h,image_w)) :
                        gy += image.getpixel((l,k+1))#(block[k+1][l])
                    
                    #hog
                    hog[0][k-i][l-j] = math.sqrt(gx**2 + gy**2)
                    hog[1][k-i][l-j] = math.degrees(math.atan2(gy,gx))
                    #mag[k][l] = hog[0][k-i][l-j]
                    #ang[k][l] = (hog[1][k-i][l-j] if hog[1][k-i][l-j] > 0 else 360 + hog[1][k-i][l-j])
            #we have mag and direction at each pixel in this block 
            #next step create histogram of bins 
            
            #bin_,angles_  = calculate_hog_simple(hog,angles_)
            bin_ = calculate_hog_weighted(hog)
            
            #normalization
            bin_ = util.normalize(bin_)
            #bin_mat[int(i/10)][int(j/30)] = bin_
            
            hog_vector[int(i/10)][int(j/30)] = list(bin_)
    
    #plot_angles(bin_mat,image)
    return hog_vector