import torchvision
import matplotlib.pyplot as plt
import numpy as np



def show_img(img_tensor):
    img = torchvision.utils.make_grid(img_tensor).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
