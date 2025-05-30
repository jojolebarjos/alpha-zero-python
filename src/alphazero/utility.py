import io

import torch

import lightning as L


def to_torchscript(model: L.LightningModule) -> bytes:
    script = model.to_torchscript()
    file = io.BytesIO()
    torch.jit.save(script, file)
    content = file.getvalue()
    return content


def from_torchscript(content: bytes) -> torch.ScriptModule:
    file = io.BytesIO(content)
    model = torch.jit.load(file)
    return model
