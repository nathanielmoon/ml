""""""
"""
Template for implementing QLearner  (c) 2015 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved
Template code for CS 4646/7646
Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.
We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.
-----do not edit anything above this line---
Student Name: Tucker Balch (replace with your name)
GT User ID: nmoon9 (replace with your User ID)
GT ID: 903755364 (replace with your GT ID)
"""


import random as rand
import time
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.
    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.4,
        gamma=0.9,
        rar=0.9,
        radr=0.9,
        dyna=0,
        verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.r_table = np.zeros((num_states, num_actions))
        self.tc_table = np.zeros((num_states, num_actions, num_states))

    def __select_action(self, s):
        return np.argmax(self.q_table[s])

    def __update_q_table(self, s, a, s_prime, r):
        a_prime = self.__select_action(s_prime)

        update = (1 - self.alpha) * self.q_table[s, a] + self.alpha * (
            r + self.gamma * self.q_table[s_prime, a_prime]
        )

        self.q_table[s, a] = update

    def __update_dyna_tables(self, s, a, s_prime, r):
        if self.dyna > 0:
            self.tc_table[s, a, s_prime] += 1
            self.r_table[s, a] = (1 - self.alpha) * self.r_table[s, a] + self.alpha * r

    def __hallucinate(self):
        if self.dyna <= 0:
            return

        s_primes = np.argmax(self.tc_table, axis=2)
        for _ in range(0, self.dyna):
            s = np.random.randint(0, self.num_states)
            a = np.random.randint(0, self.num_actions)
            s_prime = s_primes[s, a]
            r = self.r_table[s, a]
            self.__update_q_table(s, a, s_prime, r)

    def author(self):
        return "nmoon9"

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table
        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        action = self.__select_action(s)
        self.a = action
        self.s = s

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action
        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        learning_instance = (self.s, self.a, s_prime, r)
        self.__update_q_table(*learning_instance)
        self.__update_dyna_tables(*learning_instance)
        self.__hallucinate()

        if np.random.rand() < self.rar:
            action = np.random.randint(0, self.num_actions)
        else:
            action = self.__select_action(s_prime)

        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        self.rar = self.rar * self.radr
        return action

    def run(
        self,
        P,
        R,
        has_won,
        has_lost,
        n_epochs=1000,
        max_iters=1000,
        hook=None,
        W=None,
        simulate=None,
    ):
        all_rewards = np.zeros(n_epochs)
        start_pos = 0
        n = P.shape[1]
        nrange = range(n)
        signals = []
        times = []
        for epoch in range(n_epochs):
            start = time.time_ns()
            total_reward = 0
            s = start_pos
            a = self.querysetstate(s)
            step = 0
            while (
                not has_won(s, total_reward)
                and not has_lost(s, total_reward)
                and step < max_iters
            ):
                s_ = np.random.choice(nrange, p=P[a][s])
                r = None
                if len(R.shape) == 1:
                    r = R[s_]
                else:
                    r = R[s][a]
                a = self.query(s_, r)
                total_reward += r
                s = s_
                step += 1

            end = time.time_ns()
            times.append(end - start)
            if hook:
                signal = hook(
                    self,
                    simulate=simulate,
                    P=P,
                    R=R,
                    W=W,
                    variation=0.0,
                    max_steps=n * 2,
                    gamma=self.gamma,
                )
                signals.append(signal)
            all_rewards[epoch] = total_reward

        print("AVG", np.array(times).mean())
        return signals

    @property
    def policy(self):
        return np.argmax(self.q_table, axis=1)

    @property
    def V(self):
        return np.max(self.q_table, axis=1)


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
