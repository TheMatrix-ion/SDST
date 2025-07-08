import csv
import torch
from modules.single_channel_transformer import SingleChannelTransformer


def load_data(path: str) -> torch.Tensor:
    """Load CSV data as tensor with shape [batch, seq_len, 1]."""
    samples = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                samples.append([float(x) for x in row])
    tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(-1)
    return tensor


def main() -> None:
    data = load_data("data/sample.csv")
    model = SingleChannelTransformer()
    model.eval()
    with torch.no_grad():
        output = model(data)
    print("Output shape:", tuple(output.shape))
    print("First element:", output[0, 0].tolist())


if __name__ == "__main__":
    main()
