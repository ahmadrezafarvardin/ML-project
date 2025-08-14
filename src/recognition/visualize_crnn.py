# src/recognition/visualize_crnn.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import json

from crnn_model import CRNN
from dataset import MathExpressionDataset, collate_fn
from inference import ExpressionRecognizer
from torch.utils.data import DataLoader


class CRNNVisualizer:
    def __init__(self, model_path, dataset_path="dataset_extended"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dataset_path = dataset_path

        # Load model
        self.model = CRNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load recognizer
        self.recognizer = ExpressionRecognizer(model_path)

    def visualize_predictions(
        self, num_samples=8, save_path="results/recognition/visualizations"
    ):
        """Visualize predictions on validation samples"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Load validation data
        val_dataset = MathExpressionDataset(self.dataset_path, split="valid")

        # Create figure
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Select random samples
        indices = np.random.choice(
            len(val_dataset), min(num_samples, len(val_dataset)), replace=False
        )

        for idx, sample_idx in enumerate(indices):
            if idx >= num_samples:
                break

            sample = val_dataset.samples[sample_idx]

            # Load and preprocess image
            img = cv2.imread(sample["image_path"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get prediction
            pred_expr = self.recognizer.recognize(sample["image_path"])
            true_expr = sample["expression"].replace("x", "*")

            # Display
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"True: {true_expr}\nPred: {pred_expr}", fontsize=10)
            axes[idx].axis("off")

            # Add color coding for correctness
            if pred_expr == true_expr:
                axes[idx].patch.set_edgecolor("green")
                axes[idx].patch.set_linewidth(3)
            else:
                axes[idx].patch.set_edgecolor("red")
                axes[idx].patch.set_linewidth(3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/prediction_samples.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved prediction visualization to {save_path}/prediction_samples.png")

    def visualize_feature_maps(
        self, image_path, save_path="results/recognition/visualizations"
    ):
        """Visualize CNN feature maps"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Load and preprocess image
        img_tensor = self.recognizer.preprocess_image(image_path).to(self.device)

        # Fix: Ensure correct dimensions [batch, channels, height, width]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Hook to capture feature maps
        feature_maps = []

        def hook_fn(module, input, output):
            feature_maps.append(output.detach().cpu())

        # Register hooks for different CNN layers
        hooks = []
        conv_layers = []
        for name, module in self.model.cnn.named_children():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))
                conv_layers.append(name)

        # Forward pass
        with torch.no_grad():
            _ = self.model(img_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Visualize feature maps from different layers
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()

        # Original image
        original = cv2.imread(str(image_path))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Feature maps from different layers
        layer_indices = (
            [0, 2, 4] if len(feature_maps) > 4 else range(min(3, len(feature_maps)))
        )

        plot_idx = 1
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < len(feature_maps):
                fmap = feature_maps[layer_idx][0]  # First sample in batch

                # Show first 3 channels
                for j in range(min(3, fmap.shape[0])):
                    if plot_idx < len(axes):
                        axes[plot_idx].imshow(fmap[j].numpy(), cmap="viridis")
                        axes[plot_idx].set_title(f"Layer {layer_idx+1}, Channel {j+1}")
                        axes[plot_idx].axis("off")
                        plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_maps.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved feature maps to {save_path}/feature_maps.png")

    def visualize_attention_weights(
        self, save_path="results/recognition/visualizations"
    ):
        """Visualize sequence attention patterns (pseudo-attention from RNN)"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # This is a simplified visualization showing character positions
        val_dataset = MathExpressionDataset(self.dataset_path, split="valid")
        if len(val_dataset.samples) == 0:
            print("No validation samples found for attention visualization")
            return

        sample = val_dataset.samples[0]

        # Get model output
        img_tensor = self.recognizer.preprocess_image(sample["image_path"]).to(
            self.device
        )
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)  # [seq_len, batch, nclass]
            probs = torch.softmax(output, dim=2)

        # Create attention-like visualization
        seq_len = output.shape[0]
        num_classes = output.shape[2]

        plt.figure(figsize=(12, 8))

        # Plot 1: Probability distribution over sequence
        plt.subplot(2, 1, 1)
        top_probs, top_indices = torch.max(probs[:, 0, :], dim=1)
        top_probs = top_probs.cpu().numpy()
        positions = np.arange(seq_len)

        plt.bar(positions, top_probs)
        plt.xlabel("Sequence Position")
        plt.ylabel("Max Probability")
        plt.title("Character Confidence Across Sequence")
        plt.ylim(0, 1)

        # Add predicted characters
        for i, idx in enumerate(top_indices.cpu().numpy()):
            if idx != 16:  # Not blank
                char = self.model.idx_to_char.get(idx, "?")
                plt.text(i, top_probs[i] + 0.02, char, ha="center", fontsize=8)

        # Plot 2: Heatmap of probabilities
        plt.subplot(2, 1, 2)
        prob_matrix = probs[:, 0, :16].cpu().numpy().T  # Exclude blank for clarity

        import seaborn as sns

        sns.heatmap(
            prob_matrix,
            xticklabels=range(seq_len),
            yticklabels=[self.model.idx_to_char.get(i, "?") for i in range(16)],
            cmap="YlOrRd",
            cbar_kws={"label": "Probability"},
        )
        plt.xlabel("Sequence Position")
        plt.ylabel("Character Class")
        plt.title("Character Probability Distribution")

        plt.tight_layout()
        plt.savefig(f"{save_path}/sequence_attention.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved attention visualization to {save_path}/sequence_attention.png")

    def visualize_training_history(
        self,
        history_path="results/recognition/training_history.json",
        save_path="results/recognition/visualizations",
    ):
        """Create detailed training history plots"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Load history
        with open(history_path, "r") as f:
            history = json.load(f)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Loss curves
        axes[0, 0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(history["val_accuracy"], color="green", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Validation Accuracy")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # Add best accuracy marker
        best_acc_epoch = np.argmax(history["val_accuracy"])
        best_acc = history["val_accuracy"][best_acc_epoch]
        axes[0, 1].scatter(best_acc_epoch, best_acc, color="red", s=100, zorder=5)
        axes[0, 1].annotate(
            f"Best: {best_acc:.3f}",
            xy=(best_acc_epoch, best_acc),
            xytext=(best_acc_epoch + 5, best_acc - 0.05),
            arrowprops=dict(arrowstyle="->", color="red"),
        )

        # Plot 3: Levenshtein Distance
        axes[1, 0].plot(history["val_distance"], color="orange", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Average Distance")
        axes[1, 0].set_title("Average Levenshtein Distance")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Learning Rate (if available) or Loss ratio
        axes[1, 1].plot(
            np.array(history["val_loss"]) / np.array(history["train_loss"]),
            color="purple",
            linewidth=2,
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Val Loss / Train Loss")
        axes[1, 1].set_title("Overfitting Indicator")
        axes[1, 1].axhline(y=1, color="black", linestyle="--", alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/training_history_detailed.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Saved training history to {save_path}/training_history_detailed.png")

    def visualize_error_analysis(
        self, num_samples=50, save_path="results/recognition/visualizations"
    ):
        """Analyze and visualize common errors"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        val_dataset = MathExpressionDataset(self.dataset_path, split="valid")

        errors = {
            "digit_confusion": [],
            "operator_errors": [],
            "parenthesis_errors": [],
            "length_mismatch": [],
        }

        # Analyze errors
        for i in tqdm(
            range(min(num_samples, len(val_dataset))), desc="Analyzing errors"
        ):
            sample = val_dataset.samples[i]
            pred = self.recognizer.recognize(sample["image_path"])
            true = sample["expression"].replace("x", "*")

            if pred != true:
                # Categorize error
                if len(pred) != len(true):
                    errors["length_mismatch"].append((true, pred))
                else:
                    for j, (t, p) in enumerate(zip(true, pred)):
                        if t != p:
                            if t.isdigit() and p.isdigit():
                                errors["digit_confusion"].append((t, p))
                            elif t in "+-*/" or p in "+-*/":
                                errors["operator_errors"].append((t, p))
                            elif t in "()" or p in "()":
                                errors["parenthesis_errors"].append((t, p))

        # Create confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Digit confusion matrix
        if errors["digit_confusion"]:
            digit_conf = np.zeros((10, 10))
            for true_d, pred_d in errors["digit_confusion"]:
                if true_d.isdigit() and pred_d.isdigit():
                    digit_conf[int(true_d), int(pred_d)] += 1

            sns.heatmap(
                digit_conf,
                annot=True,
                fmt=".0f",
                cmap="Blues",
                xticklabels=range(10),
                yticklabels=range(10),
                ax=axes[0, 0],
            )
            axes[0, 0].set_title("Digit Confusion Matrix")
            axes[0, 0].set_xlabel("Predicted")
            axes[0, 0].set_ylabel("True")

        # Error type distribution
        error_counts = {k: len(v) for k, v in errors.items()}
        axes[0, 1].bar(error_counts.keys(), error_counts.values())
        axes[0, 1].set_title("Error Type Distribution")
        axes[0, 1].set_xlabel("Error Type")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Length distribution comparison
        true_lengths = [len(s["expression"]) for s in val_dataset.samples[:num_samples]]
        pred_lengths = []
        for s in val_dataset.samples[:num_samples]:
            pred = self.recognizer.recognize(s["image_path"])
            pred_lengths.append(len(pred))

        axes[1, 0].hist(
            [true_lengths, pred_lengths],
            label=["True", "Predicted"],
            bins=range(
                min(true_lengths + pred_lengths), max(true_lengths + pred_lengths) + 2
            ),
            alpha=0.7,
        )
        axes[1, 0].set_title("Expression Length Distribution")
        axes[1, 0].set_xlabel("Expression Length")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].legend()

        # Operator confusion
        if errors["operator_errors"]:
            ops = ["+", "-", "*", "/"]
            op_conf = np.zeros((4, 4))
            for true_op, pred_op in errors["operator_errors"]:
                if true_op in ops and pred_op in ops:
                    op_conf[ops.index(true_op), ops.index(pred_op)] += 1

            sns.heatmap(
                op_conf,
                annot=True,
                fmt=".0f",
                cmap="Oranges",
                xticklabels=ops,
                yticklabels=ops,
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Operator Confusion Matrix")
            axes[1, 1].set_xlabel("Predicted")
            axes[1, 1].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(f"{save_path}/error_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved error analysis to {save_path}/error_analysis.png")

    def visualize_model_architecture(
        self, save_path="results/recognition/visualizations"
    ):
        """Create a visual representation of the CRNN architecture"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define components
        components = [
            {"name": "Input\n64×256×1", "x": 0, "width": 1.5, "color": "lightblue"},
            {
                "name": "Conv Block 1\n64 filters",
                "x": 2,
                "width": 1.5,
                "color": "lightgreen",
            },
            {
                "name": "Conv Block 2\n128 filters",
                "x": 4,
                "width": 1.5,
                "color": "lightgreen",
            },
            {
                "name": "Conv Block 3\n256 filters",
                "x": 6,
                "width": 1.5,
                "color": "lightgreen",
            },
            {
                "name": "Conv Block 4\n512 filters",
                "x": 8,
                "width": 1.5,
                "color": "lightgreen",
            },
            {
                "name": "Feature Maps\n2×W×512",
                "x": 10,
                "width": 1.5,
                "color": "lightyellow",
            },
            {
                "name": "Bi-LSTM\n256 hidden",
                "x": 12,
                "width": 1.5,
                "color": "lightcoral",
            },
            {
                "name": "Output Layer\n17 classes",
                "x": 14,
                "width": 1.5,
                "color": "lightpink",
            },
            {"name": "CTC Decoder", "x": 16, "width": 1.5, "color": "lavender"},
            {
                "name": 'Expression\n"2+3*4"',
                "x": 18,
                "width": 1.5,
                "color": "lightgray",
            },
        ]

        # Draw components
        for i, comp in enumerate(components):
            rect = patches.Rectangle(
                (comp["x"], 2),
                comp["width"],
                4,
                linewidth=2,
                edgecolor="black",
                facecolor=comp["color"],
            )
            ax.add_patch(rect)
            ax.text(
                comp["x"] + comp["width"] / 2,
                4,
                comp["name"],
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

            # Draw arrows
            if i < len(components) - 1:
                ax.arrow(
                    comp["x"] + comp["width"],
                    4,
                    0.4,
                    0,
                    head_width=0.3,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )

        # Add annotations
        ax.text(5, 7, "CNN Feature Extraction", fontsize=12, weight="bold", ha="center")
        ax.text(13, 7, "Sequence Modeling", fontsize=12, weight="bold", ha="center")
        ax.text(17, 7, "Decoding", fontsize=12, weight="bold", ha="center")

        ax.set_xlim(-1, 20)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(
            "CRNN Architecture for Mathematical Expression Recognition",
            fontsize=16,
            weight="bold",
            pad=20,
        )

        plt.tight_layout()
        plt.savefig(f"{save_path}/model_architecture.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved architecture diagram to {save_path}/model_architecture.png")

    def visualize_data_flow(self, save_path="results/recognition/visualizations"):
        """Visualize the data flow through the model"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Get a sample
        val_dataset = MathExpressionDataset(self.dataset_path, split="valid")
        sample = val_dataset.samples[0]

        # Process through model with intermediate outputs
        img_path = sample["image_path"]
        img_tensor = self.recognizer.preprocess_image(img_path).to(self.device)

        # Fix: Ensure correct dimensions
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension if needed
        elif img_tensor.dim() == 5:
            img_tensor = img_tensor.squeeze(0)  # Remove extra dimension if present

        # Ensure shape is [batch, channels, height, width]
        if img_tensor.shape[0] != 1 or img_tensor.shape[1] != 1:
            print(f"Warning: Unexpected tensor shape: {img_tensor.shape}")
            # Try to fix common issues
            if (
                img_tensor.dim() == 5
                and img_tensor.shape[0] == 1
                and img_tensor.shape[1] == 1
            ):
                img_tensor = img_tensor.squeeze(0)

        # Get intermediate shapes
        with torch.no_grad():
            # CNN output
            cnn_out = self.model.cnn(img_tensor)
            b, c, h, w = cnn_out.size()

            # Prepare for RNN
            conv = cnn_out.permute(3, 0, 1, 2)
            conv = conv.contiguous().view(w, b, c * h)

            # RNN output
            output = self.model.rnn(conv)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Original image
        img = cv2.imread(img_path)
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Input Image\nShape: {img.shape}")
        axes[0, 0].axis("off")

        # Preprocessed tensor
        img_show = img_tensor[0, 0].cpu().numpy()
        axes[0, 1].imshow(img_show, cmap="gray")
        axes[0, 1].set_title(f"Preprocessed\nShape: {list(img_tensor.shape)}")
        axes[0, 1].axis("off")

        # CNN feature map
        feature_map = cnn_out[0, 0].cpu().numpy()
        axes[0, 2].imshow(feature_map, cmap="hot")
        axes[0, 2].set_title(f"CNN Output (1 channel)\nShape: {list(cnn_out.shape)}")
        axes[0, 2].axis("off")

        # Sequence representation
        seq_data = (
            conv[:, 0, : min(10, conv.shape[2])].cpu().numpy()
        )  # First 10 features
        axes[1, 0].imshow(seq_data.T, cmap="viridis", aspect="auto")
        axes[1, 0].set_title(f"Sequence Features\nShape: {list(conv.shape)}")
        axes[1, 0].set_xlabel("Sequence Position")
        axes[1, 0].set_ylabel("Feature Dimension")

        # Output probabilities
        probs = torch.softmax(output[:, 0, :], dim=1).cpu().numpy()
        axes[1, 1].imshow(probs.T, cmap="Blues", aspect="auto")
        axes[1, 1].set_title(f"Output Probabilities\nShape: {list(output.shape)}")
        axes[1, 1].set_xlabel("Sequence Position")
        axes[1, 1].set_ylabel("Character Class")

        # Final prediction
        pred = self.recognizer.recognize(img_path)
        axes[1, 2].text(
            0.5,
            0.5,
            f'Predicted:\n"{pred}"',
            fontsize=20,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
        )
        axes[1, 2].set_title("Final Output")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_path}/data_flow.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved data flow visualization to {save_path}/data_flow.png")

    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating all visualizations...")

        # 1. Prediction samples
        self.visualize_predictions()

        # 2. Feature maps
        val_dataset = MathExpressionDataset(self.dataset_path, split="valid")
        if len(val_dataset.samples) > 0:
            sample_path = val_dataset.samples[0]["image_path"]
            self.visualize_feature_maps(sample_path)

        # 3. Attention/sequence visualization
        self.visualize_attention_weights()

        # 4. Training history
        history_path = Path("results/recognition/training_history.json")
        if history_path.exists():
            self.visualize_training_history()

        # 5. Error analysis
        self.visualize_error_analysis()

        # 6. Model architecture
        self.visualize_model_architecture()

        # 7. Data flow
        self.visualize_data_flow()

        print("All visualizations completed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CRNN model")
    parser.add_argument(
        "--model",
        default="results/recognition/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--dataset", default="dataset_extended", help="Path to dataset")
    parser.add_argument(
        "--output",
        default="results/recognition/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="Number of samples to visualize"
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = CRNNVisualizer(args.model, args.dataset)

    # Generate all visualizations
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()
