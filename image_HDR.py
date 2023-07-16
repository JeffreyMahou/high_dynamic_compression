### IMPORTS

import numpy as np # library for scientific compuations
import matplotlib.pyplot as plt # library for plotting results
# import imageio.v2 as imageio
import imageio # reading hdr images
imageio.plugins.freeimage.download()
import argparse # parsing tool
import os # dealing with system
from scipy import interpolate # linear interpolation
from scipy.ndimage import gaussian_filter # gaussian filter
from tqdm import tqdm # charging tool


### USEFUL FUNCTIONS

def grad(H): # This function computes the gradient of an image with forward difference

    next_H_x = np.roll(H, -1, axis=1) # shift the matrix along the x axis
    next_H_x[:,-1] = H[:,-1] # border conditions : enables the gradient to be 0 on the border
    next_H_y = np.roll(H, -1, axis=0) # shift the matrix along the y axis
    next_H_y[-1] = H[-1]  # border conditions : enables the gradient to be 0 on the border

    grad_H = np.array([next_H_x - H, next_H_y - H]) # compute gradient
    # grad_H = np.gradient(H) # another possibility but the definition of the gradient doesn't match the article

    return grad_H

def depth(w, h, thresh): # Compute the depth of the gaussian pyramid

    ratios = (np.log2(w/thresh), np.log2(h/thresh)) # number of division by 2 possible before reaching threshold

    d = int(np.floor(np.min(ratios)))

    return d

def grad_k(H, k): # compute gradients of the gaussian pyramid layers

    next_H_x = np.roll(H, -1, axis=1)
    next_H_x[:,-1] = H[:,-1]

    prev_H_x = np.roll(H, 1, axis=1)
    prev_H_x[:,0] = H[:,0]

    next_H_y = np.roll(H, -1, axis=0)
    next_H_y[-1] = H[-1]

    prev_H_y = np.roll(H, 1, axis=0)
    prev_H_y[0] = H[0]

    G_x = (next_H_x - prev_H_x)/(2**(k+1)) # Here, the gradient are defined with central difference as in the article
    G_y = (next_H_y - prev_H_y)/(2**(k+1))
    grad_H = np.array([G_x, G_y])

    return grad_H

def attenuation(H, alpha, beta): # compute the attenuation of a given gradient of the gaussian pyramid

    reduced_H = H/alpha # gradient magnitude divided by a fraction of the mean gradient magnitude
    reduced_H[reduced_H==0] = 1 # the attenuation is set to 1 when the gradient is 0 because otherwise you have a division by 0
    att = reduced_H**(beta-1) # Reduce gradient as in the article

    return att

def L(Phi, new_w, new_h): # Linear interpolation

    w, h = Phi.shape[1], Phi.shape[0] # current width and height

    x = np.linspace(0, new_w-1, w) # spread the w values along x-axis ranging from 0 to new_w
    y = np.linspace(0, new_h-1, h) # spread the h values along y-axis ranging from 0 to new_h

    f = interpolate.interp2d(x, y, Phi, kind='linear') # create a function the linearly interpolates Phi

    x_new = np.arange(0, new_w)
    y_new = np.arange(0, new_h)

    return f(x_new, y_new) # Compute f along the new dimensions

def divergence(G): # Compute the divergence of a Matrix with backward difference

    prev_G_x = np.roll(G[0], 1, axis=1)
    prev_G_x[:,0] = [0]*len(G[0]) # border conditions are also set so that the gradient is 0
    prev_G_y = np.roll(G[1], 1, axis=0)
    prev_G_y[0] = [0]*len(G[0][0])

    div = G[0] - prev_G_x + G[1] - prev_G_y

    return div

def iter_poisson(I, A): # One iteration of the Poisson iterative resolution

    next_I_x = np.roll(I, -1, axis=1)
    next_I_x[:, -1] = I[:, -1]

    prev_I_x = np.roll(I, 1, axis=1)
    prev_I_x[:, 0] = I[:, 0]

    next_I_y = np.roll(I, -1, axis=0)
    next_I_y[-1] = I[-1]

    prev_I_y = np.roll(I, 1, axis=0)
    prev_I_y[0] = I[0]

    new_I = (next_I_x + prev_I_x + next_I_y + prev_I_y - A)/4

    return new_I

def rescale(image): # scale the image so that it fits into the display device : 0-1 range

    return (image-np.min(image))/(np.max(image)-np.min(image))

def show(image, title): # display a matrix as an image

    plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.show()


### MAIN CLASS


class HDR_compression():

    def __init__(self):

        self._parse()
        self._load_data()

    def _parse(self): # Parsing function

        parser = argparse.ArgumentParser(description='Computing ')
        parser.add_argument('--path', type=str, help='path to the image file', required=True) # path to the image file
        parser.add_argument('--name', type=str, help='name of the hdr file', required=True) # name of hdr file
        parser.add_argument('--save', type=int, help='if True, saves every 5 iterations to path file', default=0) # save iterations
        parser.add_argument('--step_save', type=int, help='save every step_save image', default=5) # step_save
        parser.add_argument('--alpha', type=float, help='threshold for attenuation coeffcient', default=0.01) # alpha
        parser.add_argument('--beta', type=float, help='exponent of the attenuation function', default=0.9) # beta
        parser.add_argument('--s', type=float, help="saturation of the colors", default=0.4) # saturation
        parser.add_argument('--sigma', type=float, help="standard deviation of the gaussian pyramid", default=5) # sigma
        parser.add_argument('--nb_iter', type=int, help="number of iterations of poisson process", default=5000) # number of iterations
        parser.add_argument('--multigrid', type=int, help="chose if using multigrid", default=1) # use multigrid or not

        args = parser.parse_args()
        self._path, self._name = args.path, args.name # Remark : can't load memorial.hdr
        self._save, self._step_save = bool(args.save), args.step_save
        self._alpha_coeff, self._beta, self._s, self._sigma = args.alpha, args.beta, args.s, args.sigma
        self._nb_iter = args.nb_iter
        self._multigrid = bool(args.multigrid)

    def _load_data(self): # load the image
        
        print(os.path.join(self._path, self._name))
        self.image = imageio.imread(os.path.join(self._path, self._name), format='HDR-FI')

    def compute_H(self): # compute the luminosity

        self.lum = np.sum(self.image, axis=2)/3 # luminosity : mean of the color chanels
        self.H = np.log(self.lum) # We take the logarithm to work with ratios
        self.coarse_H = self.H[::2, ::2] # coarse version of the problem (useful for the multigrid resolution of Poisson equation)

    def compute_grad_H(self): # compute the gradient of the luminosity

        self.grad_H = grad(self.H)
        self.coarse_grad_H = grad(self.coarse_H)

    def _compute_grad_H_k(self, H, thresh=32): # compute the gradients with different scales

        self.d = depth(*H.shape, thresh) # number of subgradients in the gaussian pyramid

        H_k = np.copy(H) # first gradient is the finest (similar to grad_H but with a central difference)
        grad_H_k = grad_k(H_k, 0)

        L_grad_H = [np.linalg.norm(grad_H_k, axis=0)] # List of all the gradients magnitudes

        for k in range(1, self.d+1):

            H_k = gaussian_filter(H_k, sigma=5, mode='nearest') # apply gaussian filter to blur the image
            H_k = H_k[::2,::2] # take one out of 2 pixels in each axis to downsample the Image
            grad_H_k = grad_k(H_k, k) # Take the gradient of this scaled image
            L_grad_H.append(np.linalg.norm(grad_H_k, axis=0)) # Compute the L2 norm of the gradient

        return L_grad_H

    def _compute_phi_k(self, L_grad_H): # Computes the attenuation matrices corresponding to each subgradient

        L_phi = [] # List of the attenuation matrices

        for k in range(self.d+1):

            alpha = self._alpha_coeff * np.average(L_grad_H[k]) # Alpha is a fraction of each gradient magnitude average, taking alpha to
                                                            # be dependent of k enables it to treat each gradient similarly
            phi_k = attenuation(L_grad_H[k], alpha, self._beta)
            L_phi.append(phi_k)

        return L_phi

    def compute_phi(self): # compute the final attenuation function. We notice that alpha_coeff is
                                                        # much smaller than in the article

        L_grad_H = self._compute_grad_H_k(self.H)
        coarse_L_grad_H = self._compute_grad_H_k(self.coarse_H)
        L_phi = self._compute_phi_k(L_grad_H)
        coarse_L_phi = self._compute_phi_k(coarse_L_grad_H)

        phi = L_phi[-1] # phi is initialized with the coarsest level
        coarse_phi = coarse_L_phi[-1]

        for k in range(self.d-1,-1,-1): # We backpropagate the attenuations

            h, w = L_phi[k].shape
            phi = L(phi, w , h)*L_phi[k] # Interpolation and then multiplication of the different attenuations

        for k in range(self.d-2,-1,-1): # The coarser level has one level removed

            h, w = coarse_L_phi[k].shape
            coarse_phi = L(coarse_phi, w , h)*coarse_L_phi[k]

        self.Phi = phi
        self.coarse_Phi = coarse_phi

    def compute_G(self): # Computing the new attenuated gradient

        self.G = self.Phi*self.grad_H # multiplication between the gradient and the attenuation function
        self.coarse_G = self.coarse_Phi*self.coarse_grad_H

    def divergence_G(self): # Compute the divergence of the attenuated gradient

        self.div_G = divergence(self.G)
        self.coarse_div_G = divergence(self.coarse_G)

    def _compute_new_image(self, image, lum, I, s): # Compute the new image with the new luminosity

        new_lum = np.exp(I) # exponentiate back the luminosity
        new_image = np.copy(image) # deep copy
        for i in range(3):
            new_image[:,:,i] = (new_image[:,:,i]/lum)**s * new_lum # formula of the article with s as saturation

        return new_image

    def poisson(self): # Resolve the Poisson equation

        h, w = self.div_G.shape
        I = np.random.randn(h, w) # start with random initialization following a gaussian shape

        self.L_loss = [] # List of the difference between two iterations

        for k in tqdm(range(self._nb_iter)):
            prev_I = np.copy(I) # previous iteration of the Poisson process
            I = iter_poisson(I, self.div_G) # Iteration of the Poisson process

            loss = np.linalg.norm(I-prev_I)/np.sqrt(w*h) # L2 norm of the difference of the images
            self.L_loss.append(loss)

            if self._save and k%self._step_save==0: # if bool save is True, saves every setp_save image
                new_image = self._compute_new_image(self.image, self.lum, I, self._s)
                plt.imsave(os.path.join(self._path, str(k) + ".png"), rescale(new_image))

        self.I = I # final iteration of the process

    def poisson_multigrid(self): # Multigrid version of the process that gives better results
        
        h, w = self.div_G.shape
        I = np.random.randn(h, w)
        self.L_loss = [] # List of the difference between two iterations

        print()
        print("PRECOMPUTING ON FINE GRID")
        for k in tqdm(range(int(self._nb_iter/10))): # a tenth of the iterations are done in the normal grid
            prev_I = np.copy(I) # previous step of the algorithm
            I = iter_poisson(I, self.div_G)

            loss = np.linalg.norm(I-prev_I)/np.sqrt(w*h) # L2 norm of the difference of the images
            self.L_loss.append(loss)

            if self._save and k%self._step_save==0:
                new_image = self._compute_new_image(self.image, self.lum, I, self._s)
                plt.imsave(os.path.join(self._path, str(k) + ".png"), rescale(new_image))

        small_I = I[::2, ::2] # select only even columns and rows
        small_h, small_w = small_I.shape

        print()
        print("COMPUTING ON COARSER GRID")
        for k in tqdm(range(self._nb_iter)):
            small_prev_I = np.copy(small_I)
            small_I = iter_poisson(small_I, self.coarse_div_G) # iterations with reduced matrices

            loss = np.linalg.norm(small_I-small_prev_I)/np.sqrt(small_w*small_h) # L2 norm of the difference of the images
            self.L_loss.append(loss)

            if self._save and k%self._step_save==0:
                new_image = self._compute_new_image(self.image[::2, ::2], self.lum[::2, ::2], small_I, self._s)
                plt.imsave(os.path.join(self._path, str(k+int(self._nb_iter/10)) + ".png"), rescale(new_image))

        I = L(small_I, w, h) # Linear interpolation to come back to normal size

        print()
        print("POSTCOMPUTING ON FINE GRID")
        for k in tqdm(range(int(self._nb_iter/5))): # Again a tenth of the iterations are computed with normal grid
            prev_I = np.copy(I)
            I = iter_poisson(I, self.div_G)

            loss = np.linalg.norm(I-prev_I)/np.sqrt(w*h)
            self.L_loss.append(loss)

            if self._save and k%self._step_save==0:
                new_image = self._compute_new_image(self.image, self.lum, I, self._s)
                plt.imsave(os.path.join(self._path, str(k+int(11*self._nb_iter/10)) + ".png"), rescale(new_image))

        self.I = I

    def colors(self): # produce final image

        new_image = self._compute_new_image(self.image, self.lum, self.I, self._s)

        self.new_image = rescale(new_image)

    def plot(self): # show results

        show(rescale(self.image), "Original image") # original image

        show(np.linalg.norm(self.grad_H, axis=0), "Original Gradient") # gradient of the image

        show(self.Phi, "Attenuation function") # attenuation function

        show(np.linalg.norm(self.G, axis=0), "modified gradient") # attenuated gradient

        show(self.new_image, "new image") # new image

        plt.plot(self.L_loss) # Plot the loss through iterations
        plt.yscale('log')
        plt.ylabel('L2 norm between two iterations')
        plt.xlabel('iterations')
        plt.title("Difference between two iterations in L2 norm")
        plt.show()

    def run(self):

        self.compute_H()
        self.compute_grad_H()
        self.compute_phi()
        self.compute_G()
        self.divergence_G()
        if self._multigrid:
            self.poisson_multigrid()
        else:
            self.poisson()
        self.colors()
        self.plot()

        print("The old dynamic range was : ", np.max(self.lum)/np.min(self.lum)) # Old dynamic range
        print("The new dynamic range is : ", np.max(np.exp(self.I))/np.min(np.exp(self.I))) # New dynamic range

### RUNNING 

Instance = HDR_compression()
Instance.run()











