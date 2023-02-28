import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_entities, positive_sign


class NoCenterInteraction(Interaction):
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

    def get_score(self, d, s, h, t):
        # Calculate the score of the triple

        ht = broadcast_cat([h, t], dim=-1)
        th = broadcast_cat([t, h], dim=-1)

        contextualized_pos = torch.abs(ht - torch.mul(s, th))

        is_entity_pair_within_para = torch.le(contextualized_pos, d).all(dim=-1)

        w = 2 * d + 1

        k = torch.mul(0.5 * (w - 1), (w - 1 / w))
        dist = torch.mul(contextualized_pos, w) - k

        dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w)[is_entity_pair_within_para]

        return -dist.norm(p=self.p, dim=-1)

    def preprocess_relations(self, r, tanh_map=True):
        d, s = r.tensor_split(2, dim=-1)

        d = torch.abs(d)

        if tanh_map:
            d = torch.tanh(d)
        else:
            d = torch.nn.ReLU(d)

        return d, s

    def forward(self, h, r, t):
        d, s = self.preprocess_relations(r, tanh_map=self.tanh_map)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(d, s, h, t)
