from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

from utils.colour_transforms.hsv import rgb2hsv_palettes

np.random.seed(0)
savepath = "../results/gmm_analysis"

palette_path = "../pastel_palettes.npy"
pastel_palettes = np.load(palette_path) / 255.

hsv_pastel_palettes = rgb2hsv_palettes(pastel_palettes)

# confirm correct HSV conversion. Seems to be correct.
for i in range(4):
    print(pastel_palettes[0, i*3:(i+1)*3] * 255)
    print(hsv_pastel_palettes[0, i*3:(i+1)*3] * np.asarray([360, 100, 100]))
    print("---")

#
# plot the primary colour
#
threeD = False
histo1 = False
gmm_vis = False
plot_palette_paths = True
if threeD:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(hsv_pastel_palettes[:,0], hsv_pastel_palettes[:,1], hsv_pastel_palettes[:,2])
    #for i in range(4):
    #    ax.scatter(hsv_pastel_palettes[:,(i*3)], hsv_pastel_palettes[:,(i*3)+1], hsv_pastel_palettes[:,(i*3)+2], label='colour {}'.format(i+1))
    c = 3  # 0: 2 groups
    ax.scatter(hsv_pastel_palettes[:,(c*3)], hsv_pastel_palettes[:,(c*3)+1], hsv_pastel_palettes[:,(c*3)+2], label='colour {}'.format(c+1))

    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    ax.legend()

    plt.show()
# c=0: 3 groups (by saturation)
# c=1: 3 groups (by saturation)
# c=2: 3 groups
# c=3: 3 groups


if histo1:
    c = 3
    plt.subplot(3, 1, 1)
    plt.hist(hsv_pastel_palettes[:, (c*3)], bins=100)
    plt.xlim(0, 1)
    plt.title('Hue')
    plt.subplot(3, 1, 2)
    plt.hist(hsv_pastel_palettes[:, (c * 3)+1], bins=100)
    plt.xlim(0, 1)
    plt.title('Saturation')
    plt.subplot(3, 1, 3)
    plt.hist(hsv_pastel_palettes[:, (c * 3)+2], bins=100)
    plt.xlim(0, 1)
    plt.title('Value')
    plt.show()

# c=0: seems to be 3 groups in hue, 2 groups in saturation, 1 group in value. Recommend 3*2*1=6 gaussian models
# c=1: seems 4 groups in hue, 3 groups in saturation, 1 group in value. Recommend 4*3*1=12 gaussian models
# c=2: 4 groups in hue, 3 groups in saturation, 1 group in value. Recommend 4*3*1=12 gaussian models
# c=3: 4 groups hues, 3 groups saturation, 2 groups value. Recommend 4*3*2=24 gaussian models

if gmm_vis:
    c=3
    #n_components = {0:6, 1:12, 2:12, 3:24}  # how many components to use to model each distribution
    n_components = {0: 3, 1: 3, 2: 3, 3: 3}  # how many components to use to model each distribution
    N = n_components[c]
    gmm_model = GaussianMixture(N)
    gmm_model.fit(hsv_pastel_palettes[:,(c*3):(c*3)+3])
    print("model parameters:")
    print(gmm_model.means_)
    # note: using full model, each component gets its own (n_features, n_features) covariance matrix. The diagonal of
    # each of these matrices is the variance in each dimension for that component.
    cov = gmm_model.covariances_
    means = gmm_model.means_
    variances = np.asarray([np.sqrt(np.diagonal(cov[i, :, :])) for i in range(0,N)])

    fig, axs = plt.subplots(1, 3, figsize=(14.4, 8.1))
    #plt.subplot(1, 3, 1)
    axs[0].scatter(hsv_pastel_palettes[:,(c*3)], hsv_pastel_palettes[:,(c*3)+1])
    axs[0].set_xlabel('hue')
    axs[0].set_xlim(0, 1)
    axs[0].set_ylabel('saturation')
    axs[0].set_ylim(0, 1)
    axs[0].set_title('hue vs saturation')
    for n in range(N):
        component_ellipse = Ellipse(means[n, [0,1]], width=variances[n, 0]*2, height=variances[n, 1]*2, alpha=0.5)
        axs[0].add_patch(component_ellipse)

    axs[1].scatter(hsv_pastel_palettes[:, (c * 3)], hsv_pastel_palettes[:, (c * 3) + 2])
    axs[1].set_xlabel('hue')
    axs[1].set_xlim(0, 1)
    axs[1].set_ylabel('value')
    axs[1].set_ylim(0, 1)
    axs[1].set_title('hue vs value')
    for n in range(N):
        component_ellipse = Ellipse(means[n, [0, 2]], width=variances[n, 0] * 2, height=variances[n, 2] * 2, alpha=0.5)
        axs[1].add_patch(component_ellipse)

    axs[2].scatter(hsv_pastel_palettes[:, (c * 3)+1], hsv_pastel_palettes[:, (c * 3) + 2])
    axs[2].set_xlabel('saturation')
    axs[2].set_xlim(0, 1)
    axs[2].set_ylabel('value')
    axs[2].set_ylim(0, 1)
    axs[2].set_title('saturation vs value')
    for n in range(N):
        component_ellipse = Ellipse(means[n, [1, 2]], width=variances[n, 1] * 2, height=variances[n, 2] * 2, alpha=0.5)
        axs[2].add_patch(component_ellipse)

    fig.suptitle("colour {} GMMs".format(c+1))
    #fig.set_size_inches(12, 6)
    plt.savefig(join(savepath, "{}_gmm_models.png".format(c+1)))
    plt.show()

if plot_palette_paths:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for palette in hsv_pastel_palettes:
        x = [palette[i] for i in range(0, 12, 3)]
        y = [palette[i] for i in range(1, 12, 3)]
        z = [palette[i] for i in range(2, 12, 3)]
        widths = [1, 2, 3, 4]
        ax.plot(x, y, z)
        #x = np.concatenate([np.linspace(x[i], x[i+1], num=10) for i in range(3)])
        #y = np.concatenate([np.linspace(y[i], y[i+1], num=10) for i in range(3)])
        #z = np.concatenate([np.linspace(z[i], z[i + 1], num=10) for i in range(3)])
        #widths = np.linspace(1, 64, num=30)
        #ax.scatter(x, y, z, s=widths)
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    plt.show()

    # average path
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.zeros((191, 4))
    y = np.zeros((191, 4))
    z = np.zeros((191, 4))
    for i, palette in enumerate(hsv_pastel_palettes):
        x[i, :] = [palette[i] for i in range(0, 12, 3)]
        y[i, :] = [palette[i] for i in range(1, 12, 3)]
        z[i, :] = [palette[i] for i in range(2, 12, 3)]
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    z = np.mean(z, axis=0)
    ax.plot(x, y, z)

    ax.set_xlabel('Hue')
    ax.set_xlim(0, 1)
    ax.set_ylabel('Saturation')
    ax.set_ylim(0, 1)
    ax.set_zlabel('Value')
    ax.set_zlim(0, 1)
    plt.show()

