# CS6475 - Spring 2021

import numpy as np
import scipy as sp
import cv2
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # for optional plots

import copy


""" Project 1: Seam Carving

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available in Canvas under Files:

(1) "Seam Carving for Content-Aware Image Resizing"
    Avidan and Shamir, 2007
    
(2) "Improved Seam Carving for Video Retargeting"
    Rubinstein, Shamir and Avidan, 2008
    
FORBIDDEN:
    1. OpenCV functions SeamFinder, GraphCut, and CostFunction are
    forbidden, along with any similar functions in the class environment.
    2. Metrics functions of error or similarity. These need to be coded from their
    mathematical equations.

GENERAL RULES:
    1. ALL CODE USED IN THIS ASSIGNMENT to generate images, red-seam images,
    differential images, and comparison metrics must be included in this file.
    2. YOU MAY ADD FUNCTIONS to this file, however it is your responsibility
    to ensure that the autograder accepts your submission.
    3. DO NOT CHANGE the format of this file. You may NOT change existing function
    signatures, including named parameters with defaults.
    4. YOU MAY NOT USE any library function that essentially completes
    seam carving or metric calculations for you. If you have questions on this,
    ask on Piazza.
    5. DO NOT IMPORT any additional libraries other than the ones included in the
    original Course Setup CS6475 environment.
    You should be able to complete the assignment with the given libraries.
    6. DO NOT INCLUDE code that saves, shows, prints, displays, or writes the
    image passed in, or your results. If you have code in the functions that 
    does any of these operations, comment it out before autograder runs.
    7. YOU ARE RESPONSIBLE for ensuring that your code executes properly.
    This file has only been tested in the course environment. any changes you make
    outside the areas annotated for student code must not impact the autograder
    system or your performance.
    
FUNCTIONS:
    returnYourName
    IMAGE GENERATION:
        beach_backward_removal
        dolphin_backward_insert + with redSeams=True
        dolphin_backward_5050
        bench_backward_removal + with redSeams=True
        bench_forward_removal + with redSeams=True
        car_backward_insert
        car_forward_insert
    COMPARISON METRICS:
        difference_image
        numerical_comparison
"""


def returnYourName():
    """ This function returns your name as shown on your Gradescope Account.
    """
    # WRITE YOUR CODE HERE.
    return "Francisco Javier Quiroga"
    raise NotImplementedError


# -------------------------------------------------------------------
""" IMAGE GENERATION
    Parameters and Returns are as follows for all of the removal/insert 
    functions:

    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)
    pctSeams : float
        Decimal value in range between(0. - 1.); percent of vertical seams to be
        inserted or removed.
    redSeams : boolean
        Boolean variable; True = this is a red seams image, False = no red seams
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c_new, ch) where c_new = new number of columns.
        Make sure you deal with any needed normalization or clipping, so that 
        your image array is complete on return.
"""

def beach_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Use the backward method of seam carving from the 2007 paper to remove
    50% of the vertical seams in the provided image. Do NOT hard-code the
    percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.     
    r, columns, channel = np.shape(image)
    number_of_seams_to_remove = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    for i in range(number_of_seams_to_remove):
        image, s = reduce_by_one_seam(image,i)
      
    return image
    raise NotImplementedError


def dolphin_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 8c, 8d from 2007 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8c with red seams
                                    Fig 8d without red seams
    """
    # WRITE YOUR CODE HERE.
   
    r, columns, channel = np.shape(image)
    number_of_seams_to_insert = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_insert), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
   
    for i in range(number_of_seams_to_insert):
        image2, s = reduce_by_one_seam(image2,i) 
        array_of_least_seams[:,:,i] = s
     
    new_seams = np.zeros_like(array_of_least_seams)
    new_seams2 = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    new_seams2 = transform_seams(new_seams)
    
    for n in range(number_of_seams_to_insert):
        image = insert_seam(image,new_seams2[:,:,n],redSeams)
    
    return image
    raise NotImplementedError




def dolphin_backward_5050(image, pctSeams=0.50, redSeams=False):
    """ Fig 8f from 2007 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    *****************************************************************
    IMPORTANT NOTE: this function is passed the image array from the 
    dolphin_backward_insert function in main.py
    *****************************************************************
    
    """
    # WRITE YOUR CODE HERE.
    r, columns, channel = np.shape(image)
    original_columns = columns/(pctSeams+1)
    number_of_seams_to_insert = copy.deepcopy(np.int32(np.floor(pctSeams*original_columns)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_insert), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
   
    for i in range(number_of_seams_to_insert):
        image2, s = reduce_by_one_seam(image2,i) 
        array_of_least_seams[:,:,i] = s
     
    new_seams = np.zeros_like(array_of_least_seams)
    new_seams2 = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    new_seams2 = transform_seams(new_seams)
    
    for n in range(number_of_seams_to_insert):
        image = insert_seam(image,new_seams2[:,:,n],redSeams)

     
    return image
    raise NotImplementedError


def bench_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Use the backward method of seam carving to remove
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8 backward with red seams
                                    Fig 8 backward without red seams
    """
    # WRITE YOUR CODE HERE.
    r, columns, channel = np.shape(image)
    number_of_seams_to_remove = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_remove), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
    for i in range(number_of_seams_to_remove):
        image2, s = reduce_by_one_seam(image2,i)
        array_of_least_seams[:,:,i] = s

    new_seams = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    for n in range(number_of_seams_to_remove):
        image = paint_seam(image,new_seams[:,:,n])
    if redSeams==True:
        result = image
    else:
        result = image2
    
    return result
    raise NotImplementedError
 
def bench_forward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Use the forward method of seam carving to remove
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8 forward with red seams
                                    Fig 8 forward without red seams
  """
    # WRITE YOUR CODE HERE.
    r, columns, channel = np.shape(image)
    number_of_seams_to_remove = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_remove), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
    for i in range(number_of_seams_to_remove):
        image2, s = reduce_by_one_forward_seam(image2,i)
        array_of_least_seams[:,:,i] = s

    new_seams = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    for n in range(number_of_seams_to_remove):
        image = paint_seam(image,new_seams[:,:,n])
        #cv2.imwrite(str(n)+"result.jpg",image)
    if redSeams==True:
        result = image
    else:
        result = image2
    #cv2.imwrite("result_not_dividing.jpg",result)
    return result
    raise NotImplementedError


def car_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    r, columns, channel = np.shape(image)
    number_of_seams_to_insert = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_insert), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
   
    for i in range(number_of_seams_to_insert):
        image2, s = reduce_by_one_seam(image2,i) 
        array_of_least_seams[:,:,i] = s
     
    new_seams = np.zeros_like(array_of_least_seams)
    new_seams2 = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    new_seams2 = transform_seams(new_seams)
    
    for n in range(number_of_seams_to_insert):
        image = insert_seam(image,new_seams2[:,:,n],redSeams)
    
    return image
    raise NotImplementedError


def car_forward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    r, columns, channel = np.shape(image)
    number_of_seams_to_insert = copy.deepcopy(np.int32(np.floor(columns*pctSeams)))
    array_of_least_seams = np.zeros((r,2,number_of_seams_to_insert), dtype=np.int32)
    image2 = np.zeros_like(image)
    image2 = copy.deepcopy(image[:,:,:])
   
    for i in range(number_of_seams_to_insert):
        image2, s = reduce_by_one_forward_seam(image2,i) 
        array_of_least_seams[:,:,i] = s
     
    new_seams = np.zeros_like(array_of_least_seams)
    new_seams2 = np.zeros_like(array_of_least_seams)
    new_seams = transform_seams(array_of_least_seams)
    new_seams2 = transform_seams(new_seams)
    
    for n in range(number_of_seams_to_insert):
        image = insert_seam(image,new_seams2[:,:,n],redSeams)
     
    return image
    raise NotImplementedError

# __________________________________________________________________
""" COMPARISON METRICS 
    There are two functions here, one for visual comparison support and one 
    for a quantitative metric. The """

def difference_image(result_image, comparison_image):
    """ Take two images and produce a difference image that best visually
    indicates where the two images differ in pixel values.
    
    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared
    
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c, ch) representing the difference between two
        images. Make sure you deal with any needed normalization or clipping,
        so that your image array is complete on return.
    """
    # WRITE YOUR CODE HERE.
    diff_image = result_image*1.0 - comparison_image*1.0 
    absolute_diff_image = np.absolute(diff_image)
    gray_absolute_diff_image = (absolute_diff_image[:,:,0] + absolute_diff_image[:,:,1] + absolute_diff_image[:,:,2])
    
    im = cv2.normalize(gray_absolute_diff_image, gray_absolute_diff_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.applyColorMap(im, cv2.COLORMAP_JET)
  
    return image
    raise NotImplementedError


def numerical_comparison(result_image, comparison_image):
    """ Take two images and produce one or two single-value metrics that
    numerically best indicate(s) how different or similar two images are.
    Only one metric is required, you may submit two, but no more.
    
    If your metric produces a result indicating a number of pixels,
    formulate it as a percentage of the total pixels in the image.

    ******************************************************************
    NOTE: You may not use functions that perform the whole function for you.
    Research methods, find an algorithm (equation) and implement it. You may
    use numpy array functions such as abs, sqrt, min, max, dot, .T and others
    that perform a single operation for you.
    ******************************************************************

    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared

    Returns
    -------
    value(s) : float   NOTE: you may present one or two metrics only.
        One or two single_value metric comparisons
        Return a tuple of values if you are using two metrics.
    """
    # WRITE YOUR CODE HERE.
    diff_image = result_image*1.0 - comparison_image*1.0
    absolute_diff_image = np.absolute(diff_image)
    im = (absolute_diff_image[:,:,0] + absolute_diff_image[:,:,1] + absolute_diff_image[:,:,2])


    shape = np.shape(im)
    total = shape[0]*shape[1]               # Total number of pixels
    count =np.count_nonzero(im)             # Number of non_zero pixels (any difference at all)
    percent_different = 100.0*count/total   # This is the percentage of pixels that have any difference at all


    av = np.average(im[im>0])           # Average of non-zero values
    average_difference = 100.0*av/765   # Where there is difference, this is the degree of difference on a scale from 0 to 100, averaged over the entire image
                                        # 765 is the maximum possible value that a pixel can hold. [255*3]
                                        # 765 would be maximum difference; 0 would be minimal difference

    t = (percent_different,average_difference)
   
    return t
    raise NotImplementedError

def paint_seam(image,seam):
    r, c, ch = np.shape(image)
    new_image = np.zeros((r,c,ch),dtype=np.uint8)
    for rows in range(r):   
        point = seam[rows]
        for columns in range(c):
            if (columns == point[0]):
                new_image[rows,columns,:] = (0,0,255)
            else:
                new_image[rows,columns,:] = image[rows,columns,:]   
    return new_image


def transform_seams(s):
    shape = np.shape(s)
    number_of_seams = shape[2]
    points_in_single_seam = shape[0]
    count = 0
    count2 = 0
    this_row = np.zeros((points_in_single_seam,number_of_seams),dtype=np.int64)
    new_row = np.zeros((points_in_single_seam,number_of_seams),dtype=np.int64)
    new_seam_array = np.zeros_like(s)
    new_seam_array[:,:,:] = s[:,:,:]
    
    for p in range(points_in_single_seam):
        count = 0
        count2 = 0
        for n in range(number_of_seams):
            this_row[p,n] = s[p,0,n]
            new_row[p,0] = this_row[p,0]
            if n > 0:
                count = sum(i <= this_row[p,n] for i in this_row[p,0:n])
                new_row[p,n] = this_row[p,n]+count
                count2 = sum(j == new_row[p,n] for j in new_row[p,0:n])
                new_row[p,n] = new_row[p,n]+count2
                new_seam_array[p,0,n] = new_row[p,n]
    
    return new_seam_array

def insert_seam(image, seam, red):  
    r, c, ch = np.shape(image)
  
    new_image = np.zeros((r,c+1,ch),dtype=np.uint8)
    for rows in range(r):   
        point = seam[rows]
     
        for columns in range(c):
            if columns<point[0]:
                new_image[rows,columns,:] = image[rows,columns,:]
            if (columns == point[0]) and (red==False):
                new_image[rows,columns,:] = image[rows,columns,:]
            if (columns == point[0]) and (red==True):
                new_image[rows,columns,:] = (0,0,255)
            if (columns>=point[0]) and (columns<c):
                new_image[rows,columns+1,:] = image[rows,columns,:]   
    return new_image




def find_seam(image):
    r, c, ch = np.shape(image)
    dx = cv2.Sobel(image, cv2.CV_64F,1,0)
    dy = cv2.Sobel(image, cv2.CV_64F,0,1)
    absolute_d = np.zeros((r, c), dtype=np.float64)
    absolute_d = (np.absolute(dx) + np.absolute(dy))/2
    average_map = np.zeros((r, c), dtype=np.float64)
    average_map = (absolute_d[:,:,0]+absolute_d[:,:,1]+absolute_d[:,:,2])/3
 
    average_map[:,-1] += 10 
    average_map[:,0] += 10
   
    K = int(r*0.03)
    if K%2==0:
        K+=1
    average_map = cv2.blur(average_map,(K,K))
    
    my_sum = 0
    seam = np.zeros((r,2), dtype=np.int32)
    least_seam = np.zeros((r,2), dtype=np.int32)
    array_of_sums = []
    array_of_seams = np.zeros((r,2,c), dtype=np.int32)
    
    for column in range(c):
        my_sum = 0
        shift = 0
        if (column > 0) and (column < c-2): 
            for row in range(r):
                if (row > 0):
                    min_val = min(average_map[row-1,column-1+shift],average_map[row-1,column+shift],average_map[row-1,column+1+shift])
                   
                    if min_val == average_map[row-1,column-1+shift]:
                        seam[row-1, 0] = column-1+shift          
                        seam[row-1, 1] = row-1                  
                        if (column + shift) > 1:
                            shift -= 1
                        if row == r-1:
                            seam[row, 0] = column-1+shift         
                            seam[row, 1] = row 
                    elif min_val == average_map[row-1,column+shift]:
                        seam[row-1, 0] = column+shift           
                        seam[row-1, 1] = row-1                  
                        if row == r-1:
                            seam[row, 0] = column+shift        
                            seam[row, 1] = row 
                    elif min_val == average_map[row-1,column+1+shift]:
                        seam[row-1, 0] = column+1+shift         
                        seam[row-1, 1] = row-1   
                        if (column + shift) < (c-2):
                            shift += 1                                           
                        if row == r-1:
                            seam[row, 0] = column+1+shift         
                            seam[row, 1] = row  
   
                    my_sum += np.float64(average_map[row,column+shift]) + np.float64(min_val)
        if my_sum < 1:
            my_sum = 100000
        array_of_sums.append(np.int64(my_sum/10))
        array_of_seams[:,:,column] = seam
   
    index_of_minimal_sum = array_of_sums.index(min(array_of_sums))
    least_seam = array_of_seams[:,:,index_of_minimal_sum]

    return least_seam

def find_forward_seam(image0):
    image = copy.deepcopy(image0)
    r, c, ch = np.shape(image)
  
    difference_matrix = np.zeros((r, c, 3), dtype=np.float64)
    cost_matrix = np.zeros((r, c), dtype=np.float64)
    
    image = image*1.0
    least_seam = np.zeros((r,2), dtype=np.int32)

    for row in range(r):
        if (row > 0):
            for column in range(c):
                #difference_matrix[row,column,:]=(image[row,column,0]+image[row,column,1]+image[row,column,2])/3
                if (column > 0) and (column < c-1): 
                    Cl_0 = abs(image[row, column+ 1,0] - image[row, column - 1,0]) + abs(image[row - 1, column,0] - image[row, column - 1,0])
                    Cu_0 = abs(image[row, column+ 1,0] - image[row, column - 1,0])
                    Cr_0 = abs(image[row, column+ 1,0] - image[row, column - 1,0]) + abs(image[row - 1, column,0] - image[row, column + 1,0])
                    Cl_1 = abs(image[row, column+ 1,1] - image[row, column - 1,1]) + abs(image[row - 1, column,1] - image[row, column - 1,1])
                    Cu_1 = abs(image[row, column+ 1,1] - image[row, column - 1,1])
                    Cr_1 = abs(image[row, column+ 1,1] - image[row, column - 1,1]) + abs(image[row - 1, column,1] - image[row, column + 1,1])
                    Cl_2 = abs(image[row, column+ 1,2] - image[row, column - 1,2]) + abs(image[row - 1, column,2] - image[row, column - 1,2])
                    Cu_2 = abs(image[row, column+ 1,2] - image[row, column - 1,2])
                    Cr_2 = abs(image[row, column+ 1,2] - image[row, column - 1,2]) + abs(image[row - 1, column,2] - image[row, column + 1,2])
                    difference_matrix[row,column,0] = Cl_0 + Cl_1 + Cl_2
                    difference_matrix[row,column,1] = Cu_0 + Cu_1 + Cu_2
                    difference_matrix[row,column,2] = Cr_0 + Cr_1 + Cr_2
                
                elif (column == 0): 
                    Cl_0 = abs(image[row, column+ 1,0] ) + abs(image[row - 1, column,0])
                    Cu_0 = abs(image[row, column+ 1,0] )
                    Cr_0 = abs(image[row, column+ 1,0] ) + abs(image[row - 1, column,0] - image[row, column + 1,0])
                    Cl_1 = abs(image[row, column+ 1,1] ) + abs(image[row - 1, column,1])
                    Cu_1 = abs(image[row, column+ 1,1] )
                    Cr_1 = abs(image[row, column+ 1,1] ) + abs(image[row - 1, column,1] - image[row, column + 1,1])
                    Cl_2 = abs(image[row, column+ 1,2] ) + abs(image[row - 1, column,2])
                    Cu_2 = abs(image[row, column+ 1,2] )
                    Cr_2 = abs(image[row, column+ 1,2] ) + abs(image[row - 1, column,2] - image[row, column + 1,2])
                    difference_matrix[row,column,0] = (Cl_0 + Cl_1 + Cl_2)
                    difference_matrix[row,column,1] = (Cu_0 + Cu_1 + Cu_2)
                    difference_matrix[row,column,2] = (Cr_0 + Cr_1 + Cr_2)
                elif (column == c-1): 
                    Cl_0 = abs(image[row, column - 1,0]) + abs(image[row - 1, column,0] - image[row, column - 1,0])
                    Cu_0 = abs(image[row, column - 1,0])
                    Cr_0 = abs(image[row, column - 1,0]) + abs(image[row - 1, column,0])
                    Cl_1 = abs(image[row, column - 1,1]) + abs(image[row - 1, column,1] - image[row, column - 1,1])
                    Cu_1 = abs(image[row, column - 1,1])
                    Cr_1 = abs(image[row, column - 1,1]) + abs(image[row - 1, column,1])
                    Cl_2 = abs(image[row, column - 1,2]) + abs(image[row - 1, column,2] - image[row, column - 1,2])
                    Cu_2 = abs(image[row, column - 1,2])
                    Cr_2 = abs(image[row, column - 1,2]) + abs(image[row - 1, column,2])
                    difference_matrix[row,column,0] = (Cl_0 + Cl_1 + Cl_2)
                    difference_matrix[row,column,1] = (Cu_0 + Cu_1 + Cu_2)
                    difference_matrix[row,column,2] = (Cr_0 + Cr_1 + Cr_2)
                
              
    
    for row in range(r):
        if (row > 0): 
            for column in range(c):
                if (column > 0) and (column < c-1):
                    Cl = difference_matrix[row,column,0]
                    Cu = difference_matrix[row,column,1]
                    Cr = difference_matrix[row,column,2]
                    cost_matrix[row,column] = min(cost_matrix[row-1,column-1]+Cl,cost_matrix[row-1,column]+Cu,cost_matrix[row-1,column+1]+Cr)
                elif column == 0:
                    Cu = difference_matrix[row,column,1]
                    Cr = difference_matrix[row,column,2]
                    cost_matrix[row,column] = min(cost_matrix[row-1,column]+Cu,cost_matrix[row-1,column+1]+Cr)
                elif column == c-1:
                    Cl = difference_matrix[row,column,0]
                    Cu = difference_matrix[row,column,1]
                    
                    cost_matrix[row,column] = min(cost_matrix[row-1,column-1]+Cl,cost_matrix[row-1,column]+Cu)
                    
    average = np.average(cost_matrix)
    last_row = cost_matrix[-1,:]
    min_index = np.argmin(cost_matrix[-1,:])
  
    for row in reversed(range(r)):
        point = least_seam[row,:]
        point[1] = row
        if min_index < c-2:
            this_min = min(cost_matrix[row-1,min_index-1],cost_matrix[row-1,min_index],cost_matrix[row-1,min_index+1])
        else:
            this_min = min(cost_matrix[row-1,min_index-1],cost_matrix[row-1,min_index])
   
        if row == r-1:
            point[0] = copy.deepcopy(min_index)
            
            
        else:
            if this_min == cost_matrix[row-1,min_index-1]:
                
                min_index -= 1
              
                if min_index < 1:
                    min_index = 0
            elif this_min == cost_matrix[row-1,min_index+1]:
                min_index += 1
           
                if min_index > c-2:
                    min_index = c-2
            point[0] = min_index
        if min_index > c-2:
            min_index = c-2
        least_seam[row,:] = point
   
    im = cv2.normalize(cost_matrix, cost_matrix, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    #cv2.imwrite("cost_matrix.jpg",image)
    return least_seam

def reduce_by_one_seam(image, iteration):  
    r, c, ch = np.shape(image)

    least_seam = find_seam(image)
    
    image_with_seam = np.zeros((r, c, ch), dtype=np.float64)
    image_with_seam = copy.deepcopy(image[:,:,:])
    image_with_seam[least_seam[:,1], least_seam[:,0]] = (0, 0, 255)
    mask = np.ones((r, c, ch), dtype=np.int32)
    mask[least_seam[:,1], least_seam[:,0]] = (0, 0, 0)
    mask_to_show = np.ones((r, c, ch), dtype=np.int32)
    mask_to_show[least_seam[:,1], least_seam[:,0]] = (0, 0, 0)
    mask_to_show = mask_to_show*255.
    toggle = False       
    
    new_image = np.zeros((r, c-1, ch), dtype=np.uint8)
    
    
    for i in range(r):
        toggle = False
        for j in range(c):
            if (mask[i,j,0]==0):
                toggle = True
            if (mask[i,j,0]==1) and (toggle == False):
                new_image[i,j,:] = image_with_seam[i,j,:]
            elif (mask[i,j,0]==1) and (toggle == True):
                new_image[i,j-1,:] = image_with_seam[i,j,:]
            
    return new_image, least_seam

def reduce_by_one_forward_seam(image, iteration):  
    r, c, ch = np.shape(image)
    least_seam = find_forward_seam(image)
    
    image_with_seam = np.zeros((r, c, ch), dtype=np.float64)
    image_with_seam = copy.deepcopy(image[:,:,:])
   
    image_with_seam[least_seam[:,1], least_seam[:,0]] = (0, 0, 255)
    
    mask = np.ones((r, c, ch), dtype=np.int32)
    mask[least_seam[:,1], least_seam[:,0]] = (0, 0, 0)
    mask_to_show = np.ones((r, c, ch), dtype=np.int32)
    mask_to_show[least_seam[:,1], least_seam[:,0]] = (0, 0, 0)
    mask_to_show = mask_to_show*255.
 
    toggle = False       
  
    new_image = np.zeros((r, c-1, ch), dtype=np.uint8)
    
    for i in range(r):
        toggle = False
        for j in range(c):
            if (mask[i,j,0]==0):
                toggle = True
            if (mask[i,j,0]==1) and (toggle == False):
                new_image[i,j,:] = image_with_seam[i,j,:]
            elif (mask[i,j,0]==1) and (toggle == True):
                new_image[i,j-1,:] = image_with_seam[i,j,:]

    return new_image, least_seam


if __name__ == "__main__":
    """ You may use this area for code that allows you to test your functions.
    This section will not be graded, and is optional. Comment out this section when you
    test on the autograder to avoid the chance of wasting time and attempts.
    """
    # WRITE YOUR CODE HERE
 
    pass
