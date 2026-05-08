import matplotlib.pyplot as plt
import os


class LossLogger:
    def __init__(self, save_dir="logs"):
        self.losses = {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log(self, name, value):
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)

    def plot(self):
        plt.figure()

        for name, values in self.losses.items():
            plt.plot(values, label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.save_dir}/loss_curve.png")
        plt.close()