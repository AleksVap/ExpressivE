import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_entities, positive_sign


class FunctionalInteraction(Interaction):
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

    def get_score(self, c_h, c_t, s_h, s_t, h, t):
        # Calculate the score of the triple

        c = torch.concat([c_h, c_t], dim=-1)  # centers
        s = torch.concat([s_t, s_h], dim=-1)  # slopes

        ht = broadcast_cat([h, t], dim=-1)
        th = broadcast_cat([t, h], dim=-1)

        dist = torch.abs(ht - c - torch.mul(s, th))

        return -dist.norm(p=self.p, dim=-1)

    def preprocess_relations(self, r, tanh_map=True):
        c_h, c_t, s_h, s_t = r.tensor_split(4, dim=-1)

        if tanh_map:
            c_h = torch.tanh(c_h)
            c_t = torch.tanh(c_t)

        return c_h, c_t, s_h, s_t

    def forward(self, h, r, t):
        c_h, c_t, s_h, s_t = self.preprocess_relations(r, tanh_map=self.tanh_map)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(c_h, c_t, s_h, s_t, h, t)
