

import numpy as np
import cv2 as cv2    
import os
import math
import fingerprint_enhancer as fe
from skimage.morphology import skeletonize
from scipy.spatial import distance
import random
from sklearn.preprocessing import MinMaxScaler
import galois
from crc import Calculator, Crc16
import matplotlib.pyplot as plt
import warnings
block_size=16
warnings.filterwarnings("ignore")
#Number of minutiae points to consider
r=20
#Number of chaff points to consider
s=180
def do_segmentation(img):
    shap =img.shape
    segmented_image=img.copy()
    bs=(block_size*2)
    seg_mask = np.ones(shap)
    threshold = np.var(img,axis=None)*0.1
    row,col=img.shape[0],img.shape[1]
    i,j = 0,0
    while i<row:
        while j<col:
            a,b = i+block_size,j+block_size
            x,y = min(row,a),min(col,b)
            loc_gs_var = np.var(img[i: x, j: y])
            if loc_gs_var <= threshold:
                seg_mask[i: x, j: y] = 0
            j+=block_size
        i+=block_size
    var = cv2.getStructuringElement(cv2.MORPH_RECT,(bs,bs))
    seg_mask = cv2.erode(seg_mask,var,iterations=1)   
    seg_mask = cv2.dilate(seg_mask,var,iterations=1)
    row,col=seg_mask.shape[0],seg_mask.shape[1]
    for i in range(row):
        for j in range(col):
            if seg_mask[i][j]==0:
                segmented_image[i][j]=255
    return segmented_image

def do_normalization(segmented_image):
    row,col=segmented_image.shape[0],segmented_image.shape[1]
    normalized_image = np.empty([row,col],dtype=float)
    desired_mean,desired_variance = 100.0 , 8000.0
    estimated_mean,estimated_variance=np.mean(segmented_image),np.var(segmented_image)
    for i in range(row):
        for j in range(col):
            var=(math.sqrt(math.pow(segmented_image[i][j]-estimated_mean,2)*(desired_variance/estimated_variance)))
            if(segmented_image[i][j]>estimated_mean):      
                normalized_image[i][j] = var+desired_mean 
            else:
                normalized_image[i][j] = desired_mean-var
    return normalized_image

def do_enhancement(normalized_image):
    enhanced_image = fe.enhance_Fingerprint(normalized_image)
    return enhanced_image

def do_binarization(enhanced_image):
    retval,binarized_image=cv2.threshold(enhanced_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binarized_image

def do_thinning(binarized_image):
    thinned_image = np.where(skeletonize(binarized_image/255), 0.0, 1.0)
    return thinned_image

def preprocessing(img):
    segmented_image = do_segmentation(img)
    normalized_image = do_normalization(segmented_image)
    enhanced_image = do_enhancement(normalized_image)
    binarized_image=do_binarization(enhanced_image)
    thinned_image = do_thinning(enhanced_image)

    return img,thinned_image


def ridge_orientation(img,thinned_image):
    scale=1
    delta=0
    bs = (block_size*2)+1
    G_x = cv2.Sobel(thinned_image/255, cv2.CV_64F, 0, 1, ksize=3,scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    G_y = cv2.Sobel(thinned_image/255, cv2.CV_64F, 1, 0, ksize=3,scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    dir_x = np.zeros(img.shape)
    dir_y = dir_x
    row,col=img.shape[0],img.shape[1]
    for i in range(0,row,1):
        for j in range(0,col,1):
            temp = block_size//2
            a,b,c,d = max(0, j-temp),min(j+temp,col),min(i+temp,row),max(0,i-temp)
            y = G_y[d: c, a: b]
            x = G_x[d: c, a: b]
            G_a = x**2-y**2
            dir_y[i, j] = np.sum(G_a)
            G_b = 2*x*y
            dir_x[i, j] = np.sum(G_b)
    gaussian_directions_x = cv2.GaussianBlur(dir_x, (bs,bs), 1.0)
    gaussian_directions_y = cv2.GaussianBlur(dir_y, (bs,bs), 1.0)
    orientation_map = 0.5*(np.arctan2(gaussian_directions_x, gaussian_directions_y))+(0.5*np.pi)

    return orientation_map

def crossing_number(i, j,thinned_image):
    if thinned_image[i, j] != 0.0:
        return 2.0
    else:
        sum,val=0.0,0
        pixel_values=[]
        row = [[(-1, -1),(-1, 0),(-1, 1)],[(0, 1),(1, 1),(1, 0)],[(1, -1),(0, -1),(-1, -1)]]
        offsets = row[0] + row[1] + row[2]
        for x,y in offsets:
            pixel_values.append(thinned_image[i+x,j+y])   
        while (val < 8):
            sum+=abs(pixel_values[val] - pixel_values[val+1])
            val+=1
        return (sum//2)

def false_minutiae_removal(img,thinned_image):
    seg_mask = np.ones(img.shape)
    row,col=img.shape[0],img.shape[1]
    i,j=0,0
    global_grayscale_variance = (np.var(thinned_image)*0.1)
    while i < row:
        while j < col:
            size_type_1 = i+block_size
            end_i = min(row,size_type_1)
            size_type_2 = j+block_size
            end_j = min(col,size_type_2)
            local_grayscale_variance = np.var(thinned_image[i: end_i, j: end_j])
            demo = lambda x : True if (local_grayscale_variance > global_grayscale_variance) else False
            if demo:
                seg_mask[i: end_i, j: end_j] = 1
            else:
                seg_mask[i: end_i, j: end_j] = 0.0
            j+=block_size
        i+=block_size        
    return seg_mask

def remove_strip(img,minutiae,seg_mask):
    i,j=0,0
    minutiae_points={}
    row,col=img.shape[0],img.shape[1]
    while j<col:
        start = 0
        size_a = j+block_size
        end_j = min(col, size_a)
        seg_mask[start: block_size, j: end_j] = 0.0
        seg_mask[ row-block_size:  row, j:end_j] = 0.0
        j+=block_size
    while i<row:
        start = 0
        size_b=i+block_size
        end_i = min( row,size_b)
        seg_mask[i: end_i, start: block_size] = 0.0
        seg_mask[i: end_i,  col-block_size:  col] = 0.0
        i+=block_size
    neighbourhood = [(0, 1), (0, -1), (0, 0), (1, 0), (-1, 0)]
    for j in minutiae:
        x=j[0]
        y=j[1]
        dec = False
        for i in range(5):
            dir_x,dir_y=neighbourhood[i][0]*block_size,neighbourhood[i][1]*block_size
            try:
                var_x = x + dir_x 
                var_y = y + dir_y
                if seg_mask[var_x,var_y] == 0.0:
                    dec = True
                    break
            except IndexError:
                dec = True
                break
        if dec==False:
            minutiae_points[(x, y)] =  minutiae[(x, y)]
    return minutiae_points

def cluster(minutiae_points):
    minutiae_list = list(minutiae_points.items())
    threshold = block_size/2
    cluster_found = False
    cluster_list = set()
    i=1
    while i<len(minutiae_list):
        j=0
        while j<i:
            (x2, y2), (_, _) = minutiae_list[j]
            (x1, y1), (_, _) = minutiae_list[i]
            if (distance.euclidean((x1, y1), (x2, y2)) <= threshold):
                cluster_list.add((x1, y1))
                cluster_list.add((x2, y2))
                cluster_found = True
            j+=1
        i+=1        
                                 
    if  cluster_found==False:
        return False,minutiae_points
    j=0
    while j<10:
        i=0
        while i<len(minutiae_list):
            if (x1, y1) not in cluster_list:
                for (x2, y2) in cluster_list:
                    (x1, y1), _ = minutiae_list[i]
                    if(distance.euclidean((x1, y1), (x2, y2))<=threshold):
                        cluster_list.add((x1, y1))
            i+=1
        j+=1                        
    for (x1, y1) in cluster_list:
        del  minutiae_points[(x1, y1)]
    return True,minutiae_points

def minutiae(img,thinned_image,orientation_map):
    minutiae = {}
    bs=(block_size*2)
    minutiae_image = cv2.cvtColor((255*thinned_image).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    row,col=img.shape[0],img.shape[1]
    i=1
    while i<row-1:
        j=1
        while j<col-1:
            cal = crossing_number(i, j,thinned_image)
            if cal == 1:
                minutiae[(i, j)] = (cal,orientation_map[i, j]) 
            elif cal == 3:
                minutiae[(i, j)] = (cal,orientation_map[i, j])
            j+=1
        i+=1        
    seg_mask = false_minutiae_removal(img,thinned_image)
    var = cv2.getStructuringElement(cv2.MORPH_RECT,(bs,bs))
    seg_mask = cv2.erode(seg_mask,var,iterations=1)   
    seg_mask = cv2.dilate(seg_mask,var,iterations=1)
    minutiae_points = remove_strip(img,minutiae,seg_mask)
    minutiae_in_cluster = True
    while minutiae_in_cluster:
        minutiae_in_cluster,minutiae_points = cluster(minutiae_points)
    for (x, y) in minutiae_points:
        pt, _ = minutiae_points[(x, y)]
        temp=[1,3]
        if pt == temp[0] :
            cv2.circle(minutiae_image, (y,x), radius=3, color=(255 ,255, 0), thickness=2)  
        if pt == temp[1]:
            cv2.circle(minutiae_image, (y,x), radius=3, color=(0,255,0), thickness=2)
    return minutiae_image,minutiae_points

def minutiae_points_computer(img):
    img,thinned_image= preprocessing(img)
    orientation_map=ridge_orientation(img,thinned_image)
    minutiae_image,minutiae_points=minutiae(img,thinned_image,orientation_map)
    return minutiae_points

def vault_constructor(minutiae_points,img):
    X_orig=[]
    Y_orig=[]
    Theta_orig=[]
    for key in minutiae_points.keys():
        X_orig.append(key[0])
        Y_orig.append(key[1])
    if (len(X_orig)<20):
        print("Minutiae points are not sufficient to generate a vault")
        exit
    for value in minutiae_points.values():
        Theta_orig.append(value[1])
    X_orig=np.array(X_orig[:r])
    Y_orig=np.array(Y_orig[:r])
    Theta=np.array(Theta_orig[:r])
    #Rounding
    Theta_orig=np.round(a=Theta_orig,decimals=3)*100
    Theta_orig=Theta_orig.astype(int)
    (U,V)=img.shape
    img_x=[i for i in range(0,U+1)]
    img_y=[i for i in range(0,V+1)]
    img_theta=[i for i in range(0,361)]
    img_x=np.array(img_x)
    img_y=np.array(img_y)
    img_theta=np.array(img_theta)
    output_x=np.setdiff1d(img_x,X_orig)
    output_y=np.setdiff1d(img_y,Y_orig)
    output_theta=np.setdiff1d(img_theta,Y_orig)
    X_chaff= np.random.choice(output_x, size=s, replace=False)
    Y_chaff= np.random.choice(output_y, size=s, replace=False)
    Theta_chaff=np.random.choice(output_theta,size=s,replace=False)
    B_u=6
    B_v=6
    B_theta=4
    scaler_x = MinMaxScaler(feature_range=(0, 2**B_u))
    scaler_y = MinMaxScaler(feature_range=(0, 2**B_v))
    scaler_theta = MinMaxScaler(feature_range=(0, 2**B_theta))
    X = scaler_x.fit_transform(X_orig.reshape(-1,1))
    X = np.concatenate(X)
    X=X.astype(int)
    Y = scaler_y.fit_transform(Y_orig.reshape(-1,1))
    Y = np.concatenate(Y)
    Y=Y.astype(int)
    Theta = scaler_theta.fit_transform(Theta_orig.reshape(-1,1))
    Theta = np.concatenate(Theta)
    Theta=Theta.astype(int)
    GF = galois.GF(2**16)
    key = np.random.randint(2, size=16*16,dtype='uint16')
    data = bytes(key)
    calculator = Calculator(Crc16.CCITT, optimized=True)
    crcsum=(calculator.checksum(data))
    binary_string = np.binary_repr(crcsum, width=16)
    binary_array = np.fromiter(binary_string, dtype=int)
    key_dash=np.concatenate((key,binary_array))
    integer_list = list(key_dash)
    integer_list = [str(i) for i in integer_list]
    key_final=[]
    for i in range(1,18):
        binary_string = ''.join(integer_list[16*(i-1):16*i])
        decimal = int(binary_string, 2)
        key_final.append(decimal)

    encode = galois.Poly(key_final, field=GF)

    X_encoded=encode(X)
    Y_encoded=encode(Y)
    Theta_encoded=encode(Theta)
    #stored_key=np.packbits(key)
    output_enc_x=np.setdiff1d([i for i in range(0,2**16)],X_encoded)
    output_enc_y=np.setdiff1d([i for i in range(0,2**16)],Y_encoded)
    output_enc_theta=np.setdiff1d([i for i in range(0,2**16)],Theta_encoded)
    X_encoded_chaff= np.random.choice(output_enc_x, size=s, replace=False)
    Y_encoded_chaff= np.random.choice(output_enc_y, size=s, replace=False)
    Theta_encoded_chaff=np.random.choice(output_enc_theta,size=s,replace=False)
    Vault=dict()
    vault_print_horiz=[]
    vault_print_vertical=[]
    for i in range(r):
        key=(X[i],Y[i],Theta[i])
        value=(int(X_encoded[i]),int(Y_encoded[i]),int(Theta_encoded[i]))
        vault_print_horiz.append(X[i]+Y[i]+Theta[i])
        vault_print_vertical.append(int(X_encoded[i])+int(Y_encoded[i])+int(Theta_encoded[i]))
        Vault[key]=value
    for i in range(s):
        key=(X_chaff[i],Y_chaff[i],Theta_chaff[i])
        value=(X_encoded_chaff[i],Y_encoded_chaff[i],Theta_encoded_chaff[i])
        vault_print_horiz.append(X_chaff[i]+Y_chaff[i]+Theta_chaff[i])
        vault_print_vertical.append(int(X_encoded_chaff[i])+int(Y_encoded_chaff[i])+int(Theta_encoded_chaff[i]))
        Vault[key]=value
    vault_print=[vault_print_horiz,vault_print_vertical]
    return Vault,vault_print

def plot_creator(vault_fp_print,vault_pp_print):
    x=[]
    y=[]
    x=np.concatenate((vault_fp_print[0],vault_pp_print[0]))
    y=np.concatenate((vault_fp_print[1],vault_pp_print[1]))
    return x,y

def main():
    img_fp = np.array(cv2.imread("fingerprint.tif",0))
    minutiae_points=minutiae_points_computer(img_fp)
    Vault_fp,vault_fp_print=vault_constructor(minutiae_points,img_fp)
    
    img_pp = np.array(cv2.imread("palmprint.tif",0))
    minutiae_points=minutiae_points_computer(img_pp)
    Vault_pp,vault_pp_print=vault_constructor(minutiae_points,img_pp)
    x,y=plot_creator(vault_fp_print,vault_pp_print)
    plt.scatter(x,y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Graph')
    plt.show()

if __name__=="__main__":
    main()
