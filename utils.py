
import os, torch

def save_model_state(
    model, 
    save_dir: str = "checkpoints", 
    filename: str = "final"
) -> str:
    """ 
    Saves model weights to local directory under /checkpoints folder. 
    """

    os.makedirs(
        save_dir, 
        exist_ok=True
    )

    path = os.path.join(
        save_dir, 
        filename
        )

    torch.save(
        model.state_dict(), 
        path
    )

    return path