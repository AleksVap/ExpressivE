import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_relations, preprocess_entities


class OneBandInteraction(Interaction):
    relation_shape = (
        'e'
    )
    entity_shape = (
        'd'
    )

    def __init__(self, p: int, tanh_map: bool = True):
        super().__init__()
        self.p = p  # Norm that shall be used
        self.tanh_map = tanh_map

    def get_score(self, d, c, s, h, t):
        # Calculate the score of the triple

        contextualized_pos = torch.abs(h - c - torch.mul(s, t))

        is_entity_pair_within_band = torch.le(contextualized_pos, d).all(dim=-1)

        w = 2 * d + 1

        k = torch.mul(0.5 * (w - 1), (w - 1 / w))
        dist = torch.mul(contextualized_pos, w) - k

        dist[is_entity_pair_within_band] = torch.div(contextualized_pos, w)[is_entity_pair_within_band]

        return -dist.norm(p=self.p, dim=-1)

    def preprocess_relations(self, r, tanh_map=True):
        d, c, s = r.tensor_split(3, dim=-1)

        ReLU = torch.nn.ReLU()

        d = torch.abs(d)

        if tanh_map:
            d = torch.tanh(d)
            c = torch.tanh(c)
        else:
            d = ReLU(d)

        return d, c, s

    def forward(self, h, r, t):
        d, c, s = self.preprocess_relations(r, tanh_map=self.tanh_map)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(d, c, s, h, t)
