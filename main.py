import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from gaussian_mixture_model import GaussianMixtureModel


def main():
    # Read Images
    img = mpimg.imread('im.jpg')
    img = img.astype(float)/255

    # train the model
    print("Training with EM...")
    gmm = GaussianMixtureModel(16)
    gmm.fit(img)

    # show segmented image
    seg_img, error = gmm.return_segmented_image(img)
    print(error)
    plt.imshow(seg_img)
    plt.imsave("seg_16.jpg", seg_img)
    plt.show()


if __name__ == "__main__":
    main()
