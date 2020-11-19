import matplotlib.pyplot as plt


def plot_perf(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likehood loss')
    plt.show()