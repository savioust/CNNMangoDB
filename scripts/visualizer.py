import matplotlib.pyplot as plt
import json

def load_history(file):
    with open(file, "r") as f:
        return json.load(f)

def plot_page(metrics, history, epochs, title):
    """
    metrics: lista de tuplas (key, label)
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=18)

    axs = axs.flatten()

    for i, (key, label) in enumerate(metrics):
        axs[i].plot(epochs, history[key])
        axs[i].set_title(label)
        axs[i].set_xlabel("Epoch")
        axs[i].grid(True)

    for j in range(len(metrics), 4):
        axs[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":

    alex = load_history("history_AlexNet.json")
    efficient = load_history("history_EfficientNetB0.json")

    epochs_alex = range(1, len(alex["train_loss"]) + 1)
    epochs_eff  = range(1, len(efficient["train_loss"]) + 1)

    # ------------------ ALEXNET ------------------

    plot_page(
        metrics=[
            ("train_loss", "Train Loss"),
            ("val_accuracy", "Validation Accuracy"),
            ("val_precision", "Validation Precision")
        ],
        history=alex,
        epochs=epochs_alex,
        title="AlexNet — Página 1"
    )

    plot_page(
        metrics=[
            ("val_recall", "Validation Recall"),
            ("val_f1", "Validation F1-Score")
        ],
        history=alex,
        epochs=epochs_alex,
        title="AlexNet — Página 2"
    )

    # ------------------ EFFICIENTNETB0 ------------------

    plot_page(
        metrics=[
            ("train_loss", "Train Loss"),
            ("val_accuracy", "Validation Accuracy"),
            ("val_precision", "Validation Precision")
        ],
        history=efficient,
        epochs=epochs_eff,
        title="EfficientNetB0 — Página 1"
    )

    plot_page(
        metrics=[
            ("val_recall", "Validation Recall"),
            ("val_f1", "Validation F1-Score")
        ],
        history=efficient,
        epochs=epochs_eff,
        title="EfficientNetB0 — Página 2"
    )
