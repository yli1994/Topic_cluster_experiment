import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits



import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
#%%
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
digits = load_digits()
# Random state.
RS = 20150101
# We first reorder the data points according to the handwritten numbers.
X = np.load("temp/sif_senvec.npy")
y = np.load("temp/sif_kmeans_labels.npy")

#%%
digits_proj = TSNE(random_state=RS).fit_transform(X)
#%%

def scatter(data, labels):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", np.max(labels) + 1))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(data[:, 0], data[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(np.max(labels) + 1):
        # Position of each label.
        xtext, ytext = np.median(data[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


scatter(digits_proj, y)
#plt.savefig('digits_tsne-generated.png', dpi=120)
plt.show()