import matplotlib.pyplot as plt
import IPython.display as display

plt.ion()

def plot_learning_curve(x, scores, filename):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(x, scores, label='Scores')
    plt.legend(loc='upper left')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.grid()
    plt.savefig(filename)
    display.clear_output(wait=True)
    display.display(plt.gcf())