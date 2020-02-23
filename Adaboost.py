import numpy as np
import matplotlib.pyplot as plt

xs = np.array([[-.98, 0.],
                 [-.8, -.5],
                 [-.7, -.7],
                 [-.4, .2],
                 [-.3, .7],
                 [0., .9],
                 [0.0, 0.1],
                 [.2, .4],
                 [.9, .7],
                 [.9, -.5]
                 ])
ys = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, -1])

def plot_stumps(data, labels, stumps, classifier_weights=None):
    agg_pred = np.zeros(labels.shape)
    if classifier_weights is None:
        classifier_weights = [1] * labels.size
    for s, alpha in zip(stumps, classifier_weights):
        agg_pred += alpha * s.predict(data)
    #ties assumed positive
    agg_pred = np.sign(np.sign(agg_pred) + 0.01)
    fig, axes = plt.subplots(2)
    axes[0].set_title("Actual")
    axes[0].plot(data[labels == 1, 0], data[labels == 1, 1], 'rx')
    axes[0].plot(data[labels == -1, 0], data[labels == -1, 1], 'bo')
    axes[1].set_title("Predicted")
    axes[1].plot(data[agg_pred == 1, 0], data[agg_pred == 1, 1], 'rx')
    axes[1].plot(data[agg_pred == -1, 0], data[agg_pred == -1, 1], 'bo')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


class Stump:
    def __init__(self, data, labels, weights=None):
        """
        Initialize a stump that minimizes a weighted error function
        Assume we are working over data [-1, 1]^2
        :param data: numpy array of shape (N, 2)
        :param labels: numpy array of shape (N,) consisting of values in {-1, 1}
        :param weights: numpy array of shape (N,)
        :returns None
        """
        self.threshold = None #Threshold value to measure feature with
        self.polar = 1 # To determine if feature is classified +1 or -1
        self.alpha = None
        self.feature_index = None
        n_samples, n_features = np.shape(data)
        #classifier_wts = np.full(n_samples, (1/n_samples))
        self.min_err = 0
        min_err = float('inf') #zt 
            #classifier = Stump(data, labels)
        for feature in range(2):
            f_values = np.expand_dims(data[:,feature], axis=1)
            unique_vals = np.unique(f_values)
            for threshold in unique_vals:
                pol = 1
                prediction = np.ones(np.shape(labels))
                prediction[data[:,feature] < threshold] = -1
                score = sum(weights[labels == prediction])
                err = sum(weights[labels != prediction])
                if err > 0.5:
                    err = 1-err
                    self.polar = -1
                if err < min_err:
                    self.polar = pol
                    self.threshold = threshold
                    self.feature_index = feature
                    self.min_err = err

    def predict(self, data):
        """
        Initialize a stump that minimizes a weighted error function
        Assume we are working over data [-1, 1]^2
        :param data: numpy array of shape (N, 2)
        :returns numpy array of shape (N,) containing predictions (in {1,-1})
        """
        predictions = np.ones(data.shape[0])
        #y_pred = np.zeros((data.shape[0], 1))
        neg_index = self.polar*data[:, self.feature_index] < self.polar*self.threshold
        predictions[neg_index] = -1
        #predictions = self.alpha*predictions
        #y_pred = np.sign(y_pred).flatten()
        return predictions


def adaboost(data, labels, n_classifiers):
    """
    Run the adaboost algorithm
    :param data: numpy array of shape (N, 2)
    :param labels: numpy array of shape (N,), containing values in {1,-1}
    :param n_classifiers: number of weak classifiers to learn
    :returns a tuple (classifiers, weights) consisting of a list of stumps and a numpy array of weights for the classifiers
    """
    classifiers = []
    classifier_weights = []
    n_samples, n_features = np.shape(data)
    classifier_wts = np.full(n_samples, (1/n_samples))
    n_samples = np.shape(data)[0]
    #y_pred = np.zeros((data, 1))
    for _ in range(n_classifiers):
        classifier = Stump(data, labels, classifier_wts)
        classifier.alpha = 0.5 * np.log((1-classifier.min_err)/(classifier.min_err+1e-12))
        wts = np.transpose(classifier_wts.reshape(1,n_samples)) 
        predictions = classifier.predict(wts*data)
        # Re-adjusting the weights based on results
        classifier_wts *= np.exp(-classifier.alpha*labels*predictions)
        # Normalize
        classifier_wts /= np.sum(classifier_wts)
        weights = classifier_wts
        classifiers.append(classifier)
        classifier_weights.append(weights)
        #result = (classifiers, classifier_wts.ravel())
    return classifiers, classifier_weights