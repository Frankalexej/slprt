import pickle
import matplotlib.pylab as plt
from IPython import display
import sys

# Define recorders of training hists, for ease of extension
class Recorder: 
    def __init__(self, IOPath): 
        self.record = []
        self.IOPath = IOPath

    def save(self): 
        pass
    
    def append(self, content): 
        self.record.append(content)
    
    def get(self): 
        return self.record
    

class LossRecorder(Recorder): 
    def read(self): 
        # only used by loss hists 
        with open(self.IOPath, 'rb') as f:
            self.record = pickle.load(f)
    
    def save(self): 
        with open(self.IOPath, 'wb') as file:
            pickle.dump(self.record, file)


class HistRecorder(Recorder):     
    def save(self): 
        with open(self.IOPath, "a") as txt:
            txt.write("\n".join(self.record))
    
    def print(self, content): 
        self.append(content)
        print(content)

def draw_learning_curve_and_accuracy(losses, accs, epoch=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, best_val_loss = losses
    train_accs, valid_accs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.axvline(x=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss}')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(train_accs, label='Train')
    ax2.plot(valid_accs, label='Valid')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())



def draw_progress_bar(iteration, total, bar_length=50, title=""):
    """
    Draw a text-based progress bar in the console.

    Parameters:
    - iteration (int): The current iteration.
    - total (int): The total number of iterations.
    - bar_length (int): The length of the progress bar.

    Example usage:
    for i in range(0, total + 1):
        draw_progress_bar(i, total)
    """
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r{title} [{arrow + spaces}] {int(progress * 100)}%')
    sys.stdout.flush()