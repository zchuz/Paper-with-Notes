SimCSE模型使用固定的dropout rate， 作者觉得不行， 因此提出了将固定的dropout用一个概率分布来替代，每次forward从该概率分布中采样一个值作为dropout的值（即， positive pairs两次forward也使用不同的dropout rate）.

