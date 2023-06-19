from skimage import io
import numpy as np
import random
import argparse
import helper_functions

# handles inputs using argparse
parser = argparse.ArgumentParser(description='ALL')
parser.add_argument('--input',type=str,default='./stadium.jpg',help='path to the initial image')
parser.add_argument('--output',type=str,default='output.jpg',help='path to where the final image should be written. must include proper file extension (such as .jpg)')
parser.add_argument('--k',type=int,default='4',help='number of clusters/guassians to find')
args = parser.parse_args()

def compress_image(input,output,K_GAUSSIANS):
    '''''
    Compress an image into k pixels. Remakes any picture using 
    only k colors, while maintining the image's original look. 

    Args:
        input: string, the path to the original image.

        output: string, path to the desired output.

        k: int, the number of pixels to use when remaking the image.

    Returns:
        Nothing. will have saved a new image to the output path.

    Example: 

    '''''

    # load image
    img = io.imread(args.input)
    N_PIXELS = img.shape[0] * img.shape[1] 
    ORIGINAL_SHAPE = img.shape


    # preprocessing: flatten and scale rgb values into 0-1 range
    img = img.reshape(N_PIXELS,3)
    img = img/255

    # define a data structure to hold the cluster means. shape = one per gaussian, 3 color channels
    means = np.zeros(shape=(K_GAUSSIANS,3),dtype=float)

    # select random pixels as initial cluster centers
    for i in range(K_GAUSSIANS):
        random_point = random.randint(0, N_PIXELS)
        means[i] = img[random_point][:] 

    # define a data structure to hold each pixel's assigned cluster
    assignments = np.zeros(shape=(N_PIXELS),dtype=int) 

    # assign each pixel to the closest cluster
    helper_functions.assign_clusters(img,means,assignments,N_PIXELS,K_GAUSSIANS)


    # update clusters until convergence 
    n_iterations = 3
    for i in range(n_iterations):
        helper_functions.find_centers(img,means,assignments,N_PIXELS,K_GAUSSIANS)
        helper_functions.assign_clusters(img,means,assignments,N_PIXELS,K_GAUSSIANS)

    # now we have an estimate for the means of k guassian distrubutions, by using the k_means algorithm.

    # define a data structure to hold the covariance matracies: one covariance matrix per gaussian, each covariance matrix being a 3x3.
    covariance_matracies = np.zeros(shape=(K_GAUSSIANS,3,3),dtype=float) 

    # define a data structure to hold the weights.
    weights = np.zeros(shape=K_GAUSSIANS,dtype=float)

    helper_functions.find_init_covariance(img,means,covariance_matracies,weights,assignments,N_PIXELS,K_GAUSSIANS)

    # now we have a naieve estimate of the covariacne matracies, assuming the colors are independent and
    # using assignments instead of responsibilities. We also have a naieve estimate of the gaussian weights.

    # define a data structure to store the responsibilities: a NxK matrix of floats
    responsibilities = np.zeros(shape=(N_PIXELS,K_GAUSSIANS),dtype=float)

    # now implement the e_step and m_steps: 
    THRESHOLD = 0.0001
    log_liklihood = 0
    previous_log_likelihood = 0


    for i in range(100):
        # keep track of the convergence of the gaussians
        previous_log_likelihood = log_liklihood

        # plot log_liklihoods and perform the e_steps and m_steps
        log_liklihood = helper_functions.e_step(img, means, covariance_matracies,weights,responsibilities,N_PIXELS,K_GAUSSIANS)

        log_liklihood = helper_functions.m_step(img,means,covariance_matracies,weights,responsibilities,N_PIXELS,K_GAUSSIANS)

        # evaluate wheater the gaussaians have converged:
        change = np.abs(log_liklihood - previous_log_likelihood)
        if change < THRESHOLD:
            break
    
    # now replace each pixel with the mean of its distribution
    helper_functions.replace_image(img,assignments,means)

    # process image, write to output
    img = img * 255
    img = img.reshape(ORIGINAL_SHAPE)
    img = np.uint8(img)
    io.imsave(args.output,img)
    print("saved output to \"{}\"".format(args.output))

    return None

# this code is used if the script is run directly. bassically
# it just runs the method with the inputs from argparse. 
if __name__ == "__main__":
    compress_image(args.input,args.output,args.k)
