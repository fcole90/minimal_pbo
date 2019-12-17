"""
Minimal script to show the function learning for the preference GP
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def forrester_function(x: np.ndarray) -> np.ndarray:
    return (6 * x - 2)**2 * np.sin(12 * x - 4)

def logistic_function(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.e ** (-x))

def classification_update(
        x,
        x_data,
        y_data,
        also_variance=False,
        use_bernoulli_variance=False,
        fixed_parameters=False
):

    # Selected by hand
    sigma: float = 0.6
    theta: float = 0.06
    noise: float = 0.001  # this is only considered in the variance, not used for fitting

    if type(x) is list:
        x = np.array(x)
    if type(x_data) is list:
        x_data = np.array(x_data)
    if type(y_data) is list:
        y_data = np.array(y_data)

    # We need to compose the kernel to have both the parameters:
    # https://krasserm.github.io/2018/03/19/gaussian-processes/#scikit-learn
    # The "fixed" attribute is undocumented, but noted here:
    # https://stackoverflow.com/a/53720437/2219492
    if fixed_parameters is True:
        kernel = ConstantKernel(constant_value=sigma, constant_value_bounds="fixed") \
                 * RBF(length_scale=theta, length_scale_bounds="fixed")
        gp_classifier = GaussianProcessClassifier(kernel=kernel, optimizer=None, n_jobs=-1)
    else:
        kernel = ConstantKernel(constant_value=sigma) \
                 * RBF(length_scale=theta)
        gp_classifier = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)

    # X needs to be a matrix, even as a column.
    if len(x_data.shape) < 2:
        x_data = x_data.reshape(-1, 1)
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)

    gp_classifier.fit(x_data, y_data)
    probabilities = gp_classifier.predict_proba(x)
    updated_mean = probabilities[:, 1] # Mean probability of class 1
    if also_variance is True:
        return updated_mean
    if use_bernoulli_variance is False:
        squared_mean = gp_classifier.predict_proba(x**2)[:, 1]
        updated_variance = (squared_mean - (updated_mean**2)) + noise
    else:
        updated_variance = (probabilities[:, 1] * probabilities[:, 0]) + noise
    return updated_mean, updated_variance


def main():
    N_POINTS = 100
    N_BATCH = 100
    x = np.linspace(0, 1, N_POINTS)
    y = forrester_function(x)
    x_duels = np.array([[[x_, x_prime] for x_prime in x] for x_ in x])
    print("x_duels:", x_duels)
    true_function_2D: np.ndarray = np.array([[(y[x_prime_index] - y[x_index])
                              for x_prime_index in range(len(x))]
                             for x_index in range(len(x))])
    # Assertions to be sure values are in the proper order
    # assert true_function_2D[1, 0] == forrester_function(x_due ls[1,0,1]) - forrester_function(x_duels[1,0,0])
    true_preference_2D: np.ndarray = logistic_function(true_function_2D)

    plt.title("True function: Forrester")
    plt.plot(x, y)
    # plt.show()
    plt.close()

    plt.title("True function 2D")
    plt.imshow(true_function_2D.T, cmap="jet", interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.close()

    plt.title("True preference 2D")
    plt.imshow(true_preference_2D.T, cmap="jet", interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.close()

    # Create the data for the GP classification
    x_reshaped: np.ndarray = x_duels.reshape(-1, 2)
    print("x_reshaped:", x_reshaped)
    y_reshaped: np.ndarray = true_preference_2D.flatten()

    # Assertions to be sure values are in the proper order
    # assert y_reshaped[3] == logistic_function(forrester_function(x_reshaped[3, 1])
    #                                           - forrester_function(x_reshaped[3, 0]))

    x_data = list()
    y_data = list()
    indices_data = list()
    for i in range(N_POINTS ** 2 // N_BATCH):
        indices = np.random.randint(0, len(x_reshaped), size=N_BATCH).tolist()
        indices_data += indices
        x_data += x_reshaped[indices, :].tolist()
        x_data_array = np.array(x_data)
        y_data_probs = y_reshaped[indices].tolist()

        y_data_new = (np.random.uniform(size=len(y_data_probs)) < y_data_probs) * 1
        y_data += y_data_new.tolist()
        y_data_array = np.array(y_data)
        if len(y_data_array[y_data_array > 0.5]) < 1 or len(y_data_array[y_data_array <= 0.5]) < 1:
            continue

        predict_proba, _ = classification_update(
            x=x_reshaped,
            x_data=x_data_array,
            y_data=y_data_array,
            fixed_parameters=True,
            use_bernoulli_variance=True
        )

        predict_proba_2D = predict_proba.reshape(N_POINTS, N_POINTS)

        # if i != 0 and i % 50 == 0:

        x_winners = x_data_array[y_data_array == 1, :]
        x_losers = x_data_array[y_data_array == 0, :]
        # print("x_winners:", x_winners)
        # print("x_losers:", x_losers)
        # print("y_data_array:", y_data_array)
        # print("y_data_array == 1:", y_data_array == 1)
        plt.title(f"Sampled points {len(y_data)}")
        plt.scatter(x_winners[:, 0], x_winners[:, 1], c="red", alpha=0.3, label="winners")
        plt.scatter(x_losers[:, 0], x_losers[:, 1], c="blue", alpha=0.3, label="losers")
        plt.legend()
        plt.show()
        plt.close()

        plt.title(f"Estimated preference 2D with {len(y_data)} samples")
        plt.imshow(predict_proba_2D.T, cmap="jet", interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.show()
        plt.close()

        error_2D = np.abs(true_preference_2D - predict_proba_2D)
        plt.title(f"Estimated error 2D with {len(y_data)} samples")
        plt.imshow(error_2D.T, cmap="jet", interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        plt.close()

        print("Error2D:", np.mean(error_2D))
        print("Error1D:", np.mean(np.abs(y_reshaped - predict_proba)))
        print("Training Error1D:", np.mean(np.abs(y_reshaped[indices_data] - predict_proba[indices_data])))

        copeland_score = np.mean(predict_proba_2D, axis=1)
        plt.title(f"Copeland score with {len(y_data)} samples")
        plt.plot(x, y, c="red", label="True")
        plt.plot(x, copeland_score*10, c="green", label="Soft Copeland x10")
        plt.scatter(x[np.argmax(copeland_score)], np.max(copeland_score)*10, marker="*", c="green")
        plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="red")
        plt.legend()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()

