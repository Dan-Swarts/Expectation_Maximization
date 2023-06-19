import numpy as np


# replace each pixel with the center of its cluster, reducing the total number of colors used to K.
def replace_image(img,assignments,means):
    for i in range(len(img)):
        pixel = img[i]
        assignment = assignments[i]
        center = means[assignment]
        pixel_assignment(pixel,center)

# define a function to calculate eucledian distance between two pixels
distance = lambda p1,p2: np.sqrt(np.power(p1[0]-p2[0],2)+np.power(p1[1]-p2[1],2)+np.power(p1[2]-p2[2],2))

def evaluate_log_liklihood(img,means,covariances,weights,N_PIXELS,K_GAUSSIANS):
    # evaluate the log-likelihood: 
    # Compute the PDF values for each data point and store in a giant matrix. This is to reduce the number of 
    # calls to multivariate_normal by a factor of N_PIXELS.
    pdf_values = np.zeros((N_PIXELS, K_GAUSSIANS))
    for i in range(K_GAUSSIANS):
        pdf_values[:, i] = multivariate_normal_pdf(img, means[i], covariances[i])
    # finally, multiply by the weights, take the log sum.
    log_likelihood = np.sum(np.log(np.sum(pdf_values * weights, axis=1)))

    return log_likelihood

# define a function to help sum pixels
def pixel_add(p1,p2):
    for i in range(3):
        p1[i] += p2[i]
    return None

# define a function to help find variance. Will calculate the difference squared.
def pixel_variance(acutal_color,expected_color):
    output = [0,0,0]
    for i in range(3):
        output[i] = np.power(acutal_color[i]-expected_color[i],2)
    return output

# define a function assign a pixel's value
def pixel_assignment(p1,p2):
    for i in range(3):
        p1[i] = p2[i]
    return None

# define a function to assign each pixel to a cluster: assignment is bassed on minimum distance
def assign_clusters(img,means,assignments,N_PIXELS,K_MEANS) : 
    for i in range(N_PIXELS):
        pixel = img[i]
        # initial min is arbitrarily high (the domain is only 1 unit long in any dimension)
        minimum_distance = 10
        # iterate through cluster centers to find the closest center
        for j in range(K_MEANS):
            d = distance(pixel,means[j])
            if d < minimum_distance:
                minimum_distance = d
                assignments[i] = j

# define a function to recalculate centers
def find_centers(img,means,assignments,N_PIXELS,K_MEANS) : 

    # keep count of the number of pixels in each cluster
    counts = np.zeros(shape=(K_MEANS),dtype=int)
    # erase cluster centers
    means[:] = 0

    # sum all clusters
    for i in range(N_PIXELS):
        pixel = img[i]
        assignment = assignments[i]
        center = means[assignment]
        counts[assignment] += 1
        pixel_add(center,pixel)

    # divide by count
    for i in range(K_MEANS):
        means[i] /= counts[i]

# define a function to estimate the initial covariance matracies and weights
def find_init_covariance(img,means,covariance_matracies,weights,assignments,N_PIXELS,K_MEANS):
    sum_variance = np.zeros(shape=(K_MEANS,3),dtype=float)
    counts = np.zeros(shape=K_MEANS,dtype=int)
    
    # sum the squared difference of each pixel from its cluster center
    for i in range(N_PIXELS):
        actual_color = img[i]
        assignment = assignments[i]
        expected_color = means[assignment]
        variance = pixel_variance(actual_color,expected_color)
        pixel_add(sum_variance[assignment],variance)
        counts[assignment] += 1

    # divide by count
    for i in range(K_MEANS):
        sum_variance[i] /= counts[i]
        # update diagonals of the covariance matracies
        for c in range(3):
            covariance_matracies[i][c][c] += sum_variance[i][c]
        weights[i] = counts[i] / N_PIXELS

# custom normal distribution: 

def multivariate_normal_pdf(img, mean, covariance):
    """
    Compute the probability density function of a multivariate normal distribution.

    Parameters:
    -----------
    img: array of pixels. The data

    mean: pixel. vector containing the 3 mean variables for this gaussian

    covariance: 3x3 matrix. The covariance matrix for this gaussian

    Returns:
    --------
    pdf: the evaluated pdf for this gaussain
    """

    # Compute the first term:
    first_term = 1 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(covariance))

    # Compute the exponent term
    difference = img - mean
    exponent = -0.5 * np.sum(np.dot(difference, np.linalg.inv(covariance)) * difference, axis=1)

    # Compute the probability density function
    pdf = first_term * np.exp(exponent)

    return pdf

def e_step(img, means, covariances,weights,responsibilities,N_PIXELS,K_GAUSSIANS):
    """
    Does the e step of the EM algorithm. Will calculate the responsibilities of each pixel to
    each gaussian. The responsibility of a pixel to a gaussian represents the probability that the pixel 
    belongs to that distribution.

    Parameters:
    -----------
    img: array of N colors. an array of the pixels to calculate responsibility for

    means: array of K colors. Contains the means of each gaussian

    Covariances: array of K 3x3 matracies. contains the covariance matracies of each gaussain

    weights: array of K floats. represents the expected portion of pixels belonging to each gaussian

    responsibilities: an N by K matrix of floats. holds all responsibilities. will be populated by the method.

    N_PIXELS: int. the number of pixels in the image

    K_MEANS: int. the number of gaussians 
    ------------

    returns: the log of the liklihood function
    """
    
    for i in range(K_GAUSSIANS):
        # generate the normal distibution
        ith_normal_distribution_pdf = multivariate_normal_pdf(img, means[i], covariances[i])
        # set the ith column of responsibilities equal to the probability that x belongs to that funcion times weight.
        # this is the same as evaluating the numerator for the equation seen in the March 19 lecture. 
        responsibilities[:, i] = ith_normal_distribution_pdf * weights[i]
    
    # sum the distributions. This is the same as the denominator of the equation.
    sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)

    # evaluate final responsibilities: 
    responsibilities /= sum_responsibilities
    
    return evaluate_log_liklihood(img,means,covariances,weights,N_PIXELS,K_GAUSSIANS)

# now define the m step: 
def m_step(img,means,covariances,weights,responsibilities,N_PIXELS,K_GAUSSIANS):
    """
    Does the m step of the EM algorithm. Will update the means, weights, and 
    covarience matracies of the guassians bassed on the responsibilites found 
    in the E step. 

    Parameters:
    -----------
    img: array of N colors. the data

    means: array of K colors. Contains the mean of each gaussian

    Covariances: array of K 3x3 matracies. contains the covariance matrix of each gaussain

    weights: array of K floats. represents the expected portion of pixels belong to each gaussian

    responsibilities: an N by K matrix of floats. holds all responsibilities 
    ------------

    returns: the log of the liklihood function.
    """

    for i in range(K_GAUSSIANS):

        # grab responsibilities to the ith gaussian
        responsibility = responsibilities[:, i]

        # find the sum of responsibilites to the ith gaussian 
        sum_responsibility = np.sum(responsibility)
        
        # Update mean. Shape of responsibility altered to avoid an error.
        means[i] = np.sum(responsibility[:, np.newaxis] * img, axis=0) / sum_responsibility
        
        # Update covariance matrix
        diff = img - means[i]
        covariances[i] = np.dot(responsibility * diff.T, diff) / sum_responsibility
        
        # Update weight 
        weights[i] = sum_responsibility / N_PIXELS

    return evaluate_log_liklihood(img,means,covariances,weights,N_PIXELS,K_GAUSSIANS)