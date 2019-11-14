from tripleNet_cls import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import seaborn as sns


# Define our own plot function
def scatter(x, labels, subtitle=None):
    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit.
    txts = []
    classes = np.unique(labels)
    for i in classes:
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(subtitle)


if __name__ == '__main__':

    x_train, y_train = load_mnist()
    x_train_pairs, y_p_n_labels = generate_triplet(x_train[:200],y_train[:200], ap_pairs=20, an_pairs=20)
    Anchor = x_train_pairs[:,0,:].reshape(-1,28,28,1)
    Positive = x_train_pairs[:,1,:].reshape(-1,28,28,1)
    Negative = x_train_pairs[:,2,:].reshape(-1,28,28,1)

    model = triple_model(input_shape=(28,28,1))
    model.load_weights("triplet_model_MNIST.hdf5", by_name=True)
    y_pred = model.predict([Anchor, Positive, Negative])

    # visualize
    print("visualizing...")
    tsne = TSNE()
    tsne_embeds = tsne.fit_transform(y_pred)

    scatter(tsne_embeds, y_p_n_labels[:,0], "Training Data After TNN")
