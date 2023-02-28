import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_relations, preprocess_entities


class ExpressivEInteraction(Interaction):
    relation_shape = (
        'e'
    )
    entity_shape = (
        'd',
    )

    def __init__(self, p: int, tanh_map: bool = True, min_denom: float = 0.5):
        super().__init__()
        self.p = p  # Norm that shall be used
        self.tanh_map = tanh_map
        self.min_denom = min_denom

    def get_score(self, d_h, d_t, c_h, c_t, s_h, s_t, h, t):
        # Calculate the score of the triple

        d = torch.concat([d_h, d_t], dim=-1)  # distance
        c = torch.concat([c_h, c_t], dim=-1)  # centers
        s = torch.concat([s_t, s_h], dim=-1)  # slopes

        ht = broadcast_cat([h, t], dim=-1)
        th = broadcast_cat([t, h], dim=-1)

        contextualized_pos = torch.abs(ht - c - torch.mul(s, th))

        is_entity_pair_within_para = torch.le(contextualized_pos, d).all(dim=-1)

        w = 2 * d + 1

        k = torch.mul(0.5 * (w - 1), (w - 1 / w))
        dist = torch.mul(contextualized_pos, w) - k

        dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w)[is_entity_pair_within_para]

        return -dist.norm(p=self.p, dim=-1)

    def forward(self, h, r, t):
        d_h, d_t, c_h, c_t, s_h, s_t = preprocess_relations(r, tanh_map=self.tanh_map,
                                                            min_denom=self.min_denom)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(d_h, d_t, c_h, c_t, s_h, s_t, h, t)
