
import Backgammon
import numpy as np
from tqdm import tqdm
import pickle
# from agent_dyna import NeuralNet

import agent_dyna



agent = agent_dyna.agent()
# agent.actor.theta = pickle.load(open('saved_net_one', 'rb'))
# print(agent.actor.theta)
train = True
#agent.actor.theta = pickle.load(open('saved_net_one', 'rb'))
if(train):
    agent.actor.theta = pickle.load(open('saved_net_one', 'rb'))
def main():
    ranges = 1
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0  # Collecting stats of the games
    nGames = 1000   # how many games?
    arr = np.zeros(nGames)
    for g in tqdm(range(nGames)):
        # ##Zero eligibility traces (according to psudo code)
        winner = Backgammon.play_a_game(commentary=False, net=agent, train=train)
        winners[str(winner)] += 1
        arr[g] = winner             
        if(g % 10 == 0):

            print(agent.actor.theta)
            k = winners["1"]
            print("winrate is %f" % (k / (g + 0.00000001)))
    # print(winners)
    #  Save the agent
    if(train is True):
        file_net = open('saved_net_one', 'wb')
        pickle.dump(agent.actor.theta, file_net)
        file_net.close()
    print("Out of", ranges, nGames, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")


if __name__ == '__main__':
    main()
