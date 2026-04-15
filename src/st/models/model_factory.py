import torch
from st.models.llama3 import ModelArgs
def load_model(model, path, load_dict_only=False):
    """Load a model from a checkpoint file."""
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Handle state dict keys (with potential _orig_mod prefix from torch.compile)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
        
    step = 0
    if 'step' in checkpoint:
        step = checkpoint['step']
        
    if not load_dict_only:
        model.train() 
    return model, step

model_presets = {    'llama': {'124m': ModelArgs(n_layers=12, n_heads=12, dim=768, vocab_size=50304, max_seq_len=1024, intermediate_size=4 * 768, n_kv_heads=12),
              '978m': ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=50304, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
              '1b': ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=50304, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
              '2b': ModelArgs(n_layers=36, n_heads=32, dim=2048, vocab_size=50304, max_seq_len=1024, intermediate_size=5120, n_kv_heads=8)},

    'llama-iwslt': {'124m': ModelArgs(n_layers=12, n_heads=12, dim=768, vocab_size=49157, max_seq_len=1024, intermediate_size=4 * 768, n_kv_heads=12),
                    '500m': ModelArgs(n_layers=32, n_heads=16, dim=1024, vocab_size=49157, max_seq_len=1024, intermediate_size=3072, n_kv_heads=4),
                    '978m': ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=49157, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
                    '1b': ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=49157, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
                    '2b': ModelArgs(n_layers=36, n_heads=32, dim=2048, vocab_size=49157, max_seq_len=1024, intermediate_size=5120, n_kv_heads=8),
}
}
