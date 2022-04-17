from src.models.qlearner import QLearner
from src.problems.lotr import (
    create as create_lotr,
    position_to_coords,
    actions,
    coords_to_position,
)


def move_lotr(s, a, P, R, W):
    print("MOVING", s, a)
    x, y = position_to_coords(s, W)
    dimsize = W.shape[0]
    s_ = None
    if a == actions.up:
        if x <= 0:
            return s, 0
        s_ = coords_to_position((x - 1, y), W)
    if a == actions.down:
        if x >= dimsize - 1:
            return s, 0
        s_ = coords_to_position((x + 1, y), W)
    if a == actions.left:
        if y <= 0:
            return s, 0
        s_ = coords_to_position((x, y - 1), W)
    if a == actions.right:
        if y >= dimsize - 1:
            return s, 0
        s_ = coords_to_position((x, y + 1), W)

    print()
    return s_, R[s_]


def test_lotr():
    def has_won(s, r):
        return r > 0

    def has_lost(s, r):
        return r < -99

    P, R, W = create_lotr(n=32)
    model = QLearner(num_states=len(W.flatten()), num_actions=4)
    rewards = model.run(
        P, R, W, move_lotr, has_won, has_lost, n_epochs=1000, max_iters=1000
    )
    print(model.policy)
    print(rewards)


def run():
    test_lotr()
