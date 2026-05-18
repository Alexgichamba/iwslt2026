import torch
from st.models.conformer import Conformer

m = Conformer(
    input_dim=128, num_heads=4, ffn_dim=256, num_layers=2,
    depthwise_conv_kernel_size=15, dropout=0.1,
)
m = Conformer(
    input_dim=128, num_heads=4, ffn_dim=256, num_layers=2,
    depthwise_conv_kernel_size=1,  # ← kernel=1 means no time mixing
    dropout=0.1,
)

m.eval()

x = torch.randn(2, 50, 128)
x_alt = x.clone()
x_alt[1, 30:] = 999.0
lengths = torch.tensor([50, 30])

out_a, _ = m(x,     lengths)
out_b, _ = m(x_alt, lengths)
print("max diff in unpadded region:", (out_a[1, :30] - out_b[1, :30]).abs().max().item())