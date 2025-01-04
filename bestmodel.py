import argparse
import json
import torch
from pathlib import Path
from dataset.sentinel_loader import Sentinel_Dataset
from dataset.hrrr_loader import HRRR_Dataset
from dataset.usda_loader import USDA_Dataset
from models_mmst_vit import MMST_ViT
from models_pvt_simclr import PVTSimCLR
from util import metrics


def get_args_parser():
    parser = argparse.ArgumentParser('MMST-ViT Prediction', add_help=False)

    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for prediction')
    parser.add_argument('--embed_dim', default=512, type=int, help='Embedding dimensions')
    parser.add_argument('--context_dim', default=9, type=int, help='Context dimensions for the backbone')
    parser.add_argument('--model_path', type=str, default='./output_dir/mmst_vit/best_fine_tune_model.pth',
                        help='Path to the best fine-tuned model')
    parser.add_argument('--test_data_file', type=str, default='./data/soybean_val.json',
                        help='Path to the test data JSON file')
    parser.add_argument('--root_dir', type=str, default='./data/Tiny CropNet',
                        help='Root directory for the dataset')
    parser.add_argument('--device', default='cpu', help='Device to use for prediction (e.g., "cpu" or "cuda")')
    return parser


def load_model_with_filtering(model, checkpoint_path, device):
    """
    Load the model weights while ignoring mismatched keys.
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = model.state_dict()
    pretrained_state_dict = checkpoint

    # Filter keys
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape
    }

    # Report mismatches
    unmatched_keys = set(pretrained_state_dict.keys()) - set(filtered_state_dict.keys())
    if unmatched_keys:
        print(
            f"Warning: {len(unmatched_keys)} keys in the checkpoint did not match the model architecture and were ignored.")
        print(f"Unmatched keys: {unmatched_keys}")

    # Load the filtered state_dict
    model.load_state_dict(filtered_state_dict, strict=False)
    return model


def predict(args):
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Load test datasets
    print("Loading test datasets...")
    sentinel_test = Sentinel_Dataset(args.root_dir, args.test_data_file)
    hrrr_test = HRRR_Dataset(args.root_dir, args.test_data_file)
    usda_test = USDA_Dataset(args.root_dir, args.test_data_file)

    data_loader_sentinel_test = torch.utils.data.DataLoader(sentinel_test, batch_size=1, shuffle=False, drop_last=False)
    data_loader_hrrr_test = torch.utils.data.DataLoader(hrrr_test, batch_size=1, shuffle=False, drop_last=False)
    data_loader_usda_test = torch.utils.data.DataLoader(usda_test, batch_size=1, shuffle=False, drop_last=False)

    # Initialize the backbone
    print("Initializing the backbone...")
    pvt_backbone = PVTSimCLR(
        base_model="pvt_tiny",
        out_dim=args.embed_dim,
        context_dim=args.context_dim,
        pretrained=False
    )

    # Load the MMST-ViT model with the updated backbone
    print("Loading fine-tuned model...")
    model = MMST_ViT(
        out_dim=2,
        dim=args.embed_dim,
        batch_size=args.batch_size,
        pvt_backbone=pvt_backbone
    )
    model = load_model_with_filtering(model, args.model_path, device)
    model.to(device)
    model.eval()

    # Initialize results
    true_labels = torch.empty(0).to(device)
    pred_labels = torch.empty(0).to(device)

    # Run prediction
    print("Starting predictions...")
    with torch.no_grad():
        for (x_s, x_h, x_u) in zip(data_loader_sentinel_test, data_loader_hrrr_test, data_loader_usda_test):
            try:
                # Ensure inputs are tensors of type float
                sentinel_data = x_s[0].to(device, non_blocking=True).float()
                hrrr_data_short = x_h[0].to(device, non_blocking=True).float()
                hrrr_data_long = x_h[1].to(device, non_blocking=True).float()
                usda_data = x_u[0].to(device, non_blocking=True).float()

                # Debug input shapes
                print(f"Shape before reshaping - sentinel_data: {sentinel_data.shape}")

                # Adjust reshaping to match 6D input requirements
                # [1, 6, 16, 384, 384, 3] -> [1, 6, 16, 3, 384, 384]
                sentinel_data = sentinel_data.permute(0, 1, 2, 5, 3, 4).contiguous()

                # Debug reshaped shapes
                print(f"Shape after reshaping - sentinel_data: {sentinel_data.shape}")

                # Perform prediction
                pred = model(sentinel_data, ys=hrrr_data_short, yl=hrrr_data_long)

                # Accumulate results
                true_labels = torch.cat([true_labels, usda_data], dim=0)
                pred_labels = torch.cat([pred_labels, pred], dim=0)

            except Exception as e:
                print(f"Error processing sample: {e}")

    # Evaluate metrics
    if len(true_labels) == 0 or len(pred_labels) == 0:
        print("No valid predictions were made. Skipping metrics calculation.")
        return

    true_labels = true_labels.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    # Debug shapes before metrics calculation
    print(f"True labels shape before metrics: {true_labels.shape}")
    print(f"Predicted labels shape before metrics: {pred_labels.shape}")

    # Handle metric calculation errors gracefully
    try:
        # Initialize lists to store per-dimension metrics
        rmse_list = []
        r2_list = []
        corr_list = []

        # Compute metrics for each dimension
        num_dimensions = true_labels.shape[1]
        for i in range(num_dimensions):
            y_true_dim = true_labels[:, i]
            y_pred_dim = pred_labels[:, i]

            print(f"Computing metrics for dimension {i}")
            print(f"Shape of y_true_dim: {y_true_dim.shape}")
            print(f"Shape of y_pred_dim: {y_pred_dim.shape}")

            rmse = metrics.RMSE(y_true_dim, y_pred_dim)
            r2 = metrics.R2_Score(y_true_dim, y_pred_dim)
            corr = metrics.PCC(y_true_dim, y_pred_dim)

            rmse_list.append(rmse)
            r2_list.append(r2)
            corr_list.append(corr)

            print(f"Dimension {i}: RMSE: {rmse:.4f}, R2: {r2:.4f}, Correlation: {corr:.4f}")

        # Optionally compute the average metrics
        avg_rmse = sum(rmse_list) / num_dimensions
        avg_r2 = sum(r2_list) / num_dimensions
        avg_corr = sum(corr_list) / num_dimensions

        print(f"\nAverage Metrics:\nRMSE: {avg_rmse:.4f}\nR2: {avg_r2:.4f}\nCorrelation: {avg_corr:.4f}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    # Save predictions
    predictions = {'true_labels': true_labels.tolist(), 'predicted_labels': pred_labels.tolist()}
    predictions_path = Path('./output_dir/mmst_vit/predictions.json')
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {predictions_path}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser('MMST-ViT Prediction', parents=[get_args_parser()])
    args = parser.parse_args()

    predict(args)
