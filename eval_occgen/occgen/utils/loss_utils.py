import numpy as np
import torch
import torch.nn.functional as F

import occgen.utils.constants as C
from occgen.utils.lovasz import lovasz_softmax
from occgen.utils.vae_train_utils import kl_loss


def get_ce_function(cfg=None, device=None, *args, **kwargs):
    """
    pred: batch_size, num_classes, ...
    gt: batch_size, ...,
    """
    if cfg is None or not cfg.model.ce_use_weight:
        ce_weight = None
    else:
        content = np.array(list(cfg.dataset.content.values()))
        mapping = np.array(list(cfg.dataset.learning_map.values()))
        mapped_content = np.zeros(cfg.dataset.num_classes, dtype=int)
        np.add.at(mapped_content, mapping, content)
        mapped_percentage = mapped_content / np.sum(mapped_content)
        ce_weight = np.power(np.amax(mapped_percentage) / (mapped_percentage + C.EPSILON), 1 / 3.0)
        ce_weight = torch.tensor(ce_weight).to(device).float()
    return lambda pred, gt: F.cross_entropy(pred, gt, weight=ce_weight)


def get_lovasz_function(*args, **kwargs):
    """
    pred: -1, num_classes
    gt: -1,
    """
    return lambda pred, gt: lovasz_softmax(F.softmax(pred, dim=-1), gt)


def get_kl_function(*args, **kwargs):
    """
    mus: [mu] * 6
    logvars: [logvar] * 6
    """
    return lambda mus, logvars: (sum([kl_loss(mu, logvar) for mu, logvar in zip(mus, logvars)]) /
                                 sum([np.prod(list(mu.shape)) for mu in mus]))


def get_embed_function(*args, **kwargs):
    return lambda embed: embed


loss_functions = {
    'ce': get_ce_function,
    'lovasz': get_lovasz_function,
    'kl': get_kl_function,
    'embed': get_embed_function,
}


def build_losses(model_cfg, *args, **kwargs):
    losses = dict()
    for loss_name in model_cfg.loss_weights.keys():
        losses[loss_name] = loss_functions[loss_name](*args, **kwargs)
    return losses
