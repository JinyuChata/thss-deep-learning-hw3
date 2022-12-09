import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_folder', default='outputs')
parser.add_argument('--save_folder', default='images')
# parser.add_argument('--npz_filename', default='condTrue_16x64x64x3.npz')
args = parser.parse_args()

plt.rcParams['figure.figsize'] = (10.0, 8.0) # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def main():
    images_list = [[] for i in range(16)]

    for step in ['400', '600', '800', '1000']:
        npz_filename = f'condTrue_16x64x64x3_{step}.npz'

        # load from your saved npz
        path = f'{args.load_folder}/{npz_filename}'
        data = np.load(path)
        print(data['arr_0'].shape)

        # arr_0 is for the images, and arr_0 is for the class index
        for i, img in enumerate(data['arr_0']):
            images_list[i].append(img)
            # plt.imshow(img)
            # plt.savefig(f"{args.save_folder}/image_sample0_step{step}_{i}.png")
        for i, data in enumerate(data['arr_1']):
            print(f'figure {i}; class index:{data}')

    for image_num, images in enumerate(images_list):
        sqrtn = int(np.ceil(np.sqrt(len(images))))
        sqrtimg = 64

        fig = plt.figure(figsize=(sqrtn, sqrtn))
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape([sqrtimg, sqrtimg, 3]))
        plt.savefig(f"./images/{image_num}.png")



if __name__=='__main__':
    main()