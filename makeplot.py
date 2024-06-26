import matplotlib.pyplot as plt

def make_plot_and_save(train_losses, val_losses, EPOCHS , FOLD, Q_NUM, q_type) :
    plt.plot(range(EPOCHS) ,train_losses, label = "train_loss")
    plt.plot(range(EPOCHS),val_losses, label = "val_loss")
    plt.title(f"Loss Graph @ Q{Q_NUM+1} , FOLD{FOLD}")
    plt.ylabel("Loss")
    plt.xlabel("EPOCHS")
    plt.legend()
    plt.savefig(f"./figs/loss_{q_type}_{Q_NUM+1}_FOLD{FOLD}")
    plt.cla()
    plt.clf()
    plt.close()