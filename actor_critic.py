#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
an example of an intelligent agent who flips the board
"""
import numpy as np
import Backgammon
import torch
from torch.autograd import Variable
from tqdm import tqdm
from numpy.random import choice
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class agent():
    def __init__(self):
        self.device = torch.device('cpu')
        self.n=312
        self.w1 = Variable(0.001*torch.randn(156, self.n, device=self.device , dtype=torch.double), requires_grad=True)
        self.b1 = Variable(torch.zeros((156, 1), device=self.device , dtype=torch.double), requires_grad=True)       
        self.W = Variable(0.001*torch.randn(1, 156, device=self.device , dtype=torch.double), requires_grad=True)
        self.B = Variable(torch.zeros((1, 1), device=self.device , dtype=torch.double), requires_grad=True)
        self.theta=Variable(0.001*torch.ones(1, 156, device=self.device , dtype=torch.double), requires_grad=True)
            
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.double)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.double)
        self.Z_w1F = torch.zeros(self.w1.size(), device=self.device, dtype=torch.double)
        self.Z_b1F = torch.zeros(self.b1.size(), device=self.device, dtype=torch.double)
        self.Z_W = torch.zeros(self.W.size(), device = self.device, dtype = torch.double)
        self.Z_B = torch.zeros(self.B.size(), device = self.device, dtype = torch.double)
        self.Z_WF = torch.zeros(self.W.size(), device = self.device, dtype = torch.double)
        self.Z_BF = torch.zeros(self.B.size(), device = self.device, dtype = torch.double)
        
        self.Z_theta=torch.zeros(self.theta.size(), device = self.device, dtype = torch.double)
        self.Z_thetaF=torch.zeros(self.theta.size(), device = self.device, dtype = torch.double)
        
        self.xold=np.zeros(29)
        self.xoldF=np.zeros(29)
        self.xtheta=0
        self.xthetaF=0
        self.hidden=0
        
        self.gamma=1
        self.lam=0.7
        
        self.alpha1=0.01
        self.alpha2=0.01
        self.alphaA=0.00001
        
    def zero_el(self):
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.double)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.double)
        self.Z_w1F = torch.zeros(self.w1.size(), device=self.device, dtype=torch.double)
        self.Z_b1F = torch.zeros(self.b1.size(), device=self.device, dtype=torch.double)

        self.Z_W = torch.zeros(self.W.size(), device = self.device, dtype = torch.double)
        self.Z_B = torch.zeros(self.B.size(), device = self.device, dtype = torch.double) 
        self.Z_WF = torch.zeros(self.W.size(), device = self.device, dtype = torch.double)
        self.Z_BF = torch.zeros(self.B.size(), device = self.device, dtype = torch.double) 
        
        self.Z_theta=torch.zeros(self.theta.size(), device = self.device, dtype = torch.double)
        self.Z_thetaF=torch.zeros(self.theta.size(), device = self.device, dtype = torch.double)
        
    def update(self, player, R, board, isGameOver):
        if (player==1 | isGameOver):
            if isGameOver:
                target=0
                reward=R
                x = Variable(torch.tensor(oneHot(board), dtype = torch.double, device = self.device)).view(self.n,1)
                h = torch.mm(self.w1,x) + self.b1
                h_sigmoid = h.sigmoid() 
                hidden=h_sigmoid
            else:
                x = Variable(torch.tensor(oneHot(board), dtype = torch.double, device = self.device)).view(self.n,1)
                h = torch.mm(self.w1,x) + self.b1 
                h_sigmoid = h.sigmoid()
                hidden=h_sigmoid
                y = torch.mm(self.W,h_sigmoid) + self.B 
                y_sigmoid = y.sigmoid() 
                target = y_sigmoid.detach().cpu().numpy()
                reward = 0.0
            
            x = Variable(torch.tensor(oneHot(self.xold), dtype = torch.double, device = self.device)).view(self.n,1)
            h = torch.mm(self.w1,x) + self.b1 
            h_sigmoid = h.sigmoid() 
            y = torch.mm(self.W,h_sigmoid) + self.B
            y_sigmoid = y.sigmoid() 
            oldtarget = y_sigmoid.detach().cpu().numpy()
            
            delta=reward+self.gamma*target-oldtarget
            
            y_sigmoid.backward()
            
            self.Z_w1 = self.gamma * self.lam * self.Z_w1 + self.w1.grad.data
            self.Z_b1 = self.gamma * self.lam * self.Z_b1 + self.b1.grad.data
            self.Z_W = self.gamma * self.lam * self.Z_W + self.W.grad.data
            self.Z_B = self.gamma * self.lam * self.Z_B + self.B.grad.data
            
            self.w1.grad.data.zero_()
            self.b1.grad.data.zero_()
            self.W.grad.data.zero_()
            self.B.grad.data.zero_()
            
            delta =  torch.tensor(delta, dtype = torch.double, device = self.device)
            self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
            self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
            self.W.data = self.W.data + self.alpha2 * delta * self.Z_W
            self.B.data = self.B.data + self.alpha2 * delta * self.Z_B
            
            if player==1:
                
                grad_ln_pi = hidden - self.xtheta
               # print(grad_ln_pi)
                self.Z_theta=self.gamma * self.lam * self.Z_theta + grad_ln_pi.view(1,len(grad_ln_pi))
                self.theta.data =self.theta + self.alphaA*delta*self.Z_theta
        
        if (player==-1 | isGameOver):
            if isGameOver:
                target=0
                rewardF=1-R
                x_flipped = Variable(torch.tensor(oneHot(flip_board(board)), dtype = torch.double, device = self.device)).view(self.n,1)
                h = torch.mm(self.w1,x_flipped) + self.b1  
                h_sigmoid = h.sigmoid() 
                hidden=h_sigmoid
            else:
                x_flipped = Variable(torch.tensor(oneHot(flip_board(board)), dtype = torch.double, device = self.device)).view(self.n,1)
                h = torch.mm(self.w1,x_flipped) + self.b1 
                h_sigmoid = h.sigmoid()
                hidden=h_sigmoid
                y = torch.mm(self.W,h_sigmoid) + self.B 
                y_sigmoid = y.sigmoid() 
                target = y_sigmoid.detach().cpu().numpy()
                rewardF = 0.0
            
            x = Variable(torch.tensor(oneHot(flip_board(self.xoldF)), dtype = torch.double, device = self.device)).view(312,1)
            h = torch.mm(self.w1,x) + self.b1 
            h_sigmoid = h.sigmoid()
            y = torch.mm(self.W,h_sigmoid) + self.B 
            y_sigmoid = y.sigmoid() 
            oldtarget = y_sigmoid.detach().cpu().numpy()
            
            delta=rewardF+self.gamma*target-oldtarget
            
            y_sigmoid.backward()
            
            self.Z_w1F = self.gamma * self.lam * self.Z_w1F + self.w1.grad.data
            self.Z_b1F = self.gamma * self.lam * self.Z_b1F + self.b1.grad.data
            self.Z_WF = self.gamma * self.lam * self.Z_WF + self.W.grad.data
            self.Z_BF = self.gamma * self.lam * self.Z_BF + self.B.grad.data
            
            self.w1.grad.data.zero_()
            self.b1.grad.data.zero_()
            self.W.grad.data.zero_()
            self.B.grad.data.zero_()
            
            delta =  torch.tensor(delta, dtype = torch.double, device = self.device)
            self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1F
            self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1F
            self.W.data = self.W.data + self.alpha2 * delta * self.Z_WF
            self.B.data = self.B.data + self.alpha2 * delta * self.Z_BF
            
            if player==-1:
                grad_ln_pi = hidden - self.xtheta
                self.Z_thetaF=self.gamma*self.lam*self.Z_thetaF+ grad_ln_pi.view(1,len(grad_ln_pi))
                self.theta.data =self.theta + self.alphaA*delta*self.Z_thetaF
        
    def greedy_action(self,board, dice, player, i):
        if player == -1: board = flip_board(board)
            
        # check out the legal moves available for the throw
        possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player=1)
        
        # if there are no moves available, return an empty move
        if len(possible_moves) == 0: 
            return [] 
        
        na=len(possible_boards)
        enc=np.zeros((na, 312))
        for i in range(0, na):
            enc[i,:] = oneHot(possible_boards[i])
        x = Variable(torch.tensor(enc.transpose(), dtype = torch.double, device = self.device))
        
        h = torch.mm(self.w1,x) + self.b1 
        h_sigmoid = h.sigmoid()
        y = torch.mm(self.W,h_sigmoid) + self.B 
        va = y.sigmoid().detach().cpu()
        action = possible_moves[np.argmax(va)]
        
        if player == -1: action=flip_move(action)
        
        return action
    
    def softMax(self, board, dice, player, i):
        if player == -1: board = flip_board(board)
            
        # check out the legal moves available for the throw
        possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player=1)
        
        # if there are no moves available, return an empty move
        if len(possible_moves) == 0: 
            return [] 
        
        na=len(possible_boards)
        enc=np.zeros((na, 312))
        for i in range(0, na):
            enc[i,:] = oneHot(possible_boards[i])
        x = Variable(torch.tensor(enc.transpose(), dtype = torch.double, device = self.device))
        h = torch.mm(self.w1,x) + self.b1
        h_sigmoid = h.sigmoid() 
        pi = (torch.mm(self.theta,h_sigmoid)).softmax(1)
        xtheta_mean = torch.sum(torch.mm(h_sigmoid,torch.diagflat(pi)),1)
        xtheta_mean = torch.unsqueeze(xtheta_mean,1)
        if player==1:  
            self.xtheta=xtheta_mean
        else:
            self.xthetaF=xtheta_mean
        self.xtheta=xtheta_mean
        
        
        m = torch.multinomial(pi, 1)

        action=possible_moves[m]
        
        
        if player == -1: action=flip_move(action)
        
        return action

def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
        
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move

def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # starts by flipping the board so that the player always sees himself as player 1
    if player == -1: board_copy = flip_board(board_copy)
        
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)
    
    # if there are no moves available, return an empty move
    if len(possible_moves) == 0: 
        return [] 
    
    # Make the bestmove:
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    move = possible_moves[np.random.randint(len(possible_moves))]
    
    # if the table was flipped the move has to be flipped as well
    if player == -1: move = flip_move(move)
    
    return move




def learnit(numGames, agent):
    numWins=[]
    for g in tqdm(range(numGames)): 
        if g%1000==0:
            #print(agent.theta)
            wins=compete(agent)
            numWins.append(wins)
           
        board=Backgammon.init_board()
        
        agent.zero_el()
        if (0 == np.random.randint(2)):
            player = 1 
        else:
            player = -1
        
        moveNr=0
        isGameOver=False
        
        while(isGameOver==False):
            dice = Backgammon.roll_dice()
            for repeat in range(1+int(dice[0] == dice[1])):

                action = agent.greedy_action(np.copy(board), dice, player, repeat)
            for i in range(0,len(action)):
                board = Backgammon.update_board(board, action[i], player)
            
            R=0
            if (1 == Backgammon.game_over(board)):
                if (player == 1): 
                    R = 1.0
                else:
                    R = 0
                isGameOver = True
            if ((1 < moveNr) & (len(action) > 0)):
                agent.update(player,R, board, isGameOver)
            
            if (len(action)>0):
                if player==1:
                    agent.xold=board
                else:
                    agent.xoldF=flip_board(board)
            player=-player
            moveNr+=1
    
    x=np.arange(0, numGames, 1000)
    fig = plt.figure()
    #plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Number of games")
    ax.set_ylabel("Wins against a random player")
    ax.plot(x, numWins)

def compete(agent):
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0
    for g in range(100):
        
        board = Backgammon.init_board() 
        
        if (0 == np.random.randint(2)):
            player = 1 
        else:
            player = -1
            
        isGameOver=False
        while (isGameOver==False):
            dice = Backgammon.roll_dice()
            for repeat in range(1+int(dice[0] == dice[1])):
                if (player == -1):
                    action = Backgammon.random_agent(np.copy(board), dice, player, repeat)
                else: 
                    action = agent.greedy_action(np.copy(board), dice, player, repeat)
                for i in range(0,len(action)):
                    board = Backgammon.update_board(board, action[i], player)
            if (1 == Backgammon.game_over(board)): 
                winner = player
                isGameOver = True
                break;
            player=-player
        winners[str(winner)] += 1
   # numWins.append(winners["1"])        
    print("Out of", 100, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")
    return winners["1"]

def oneHot(board):
    oneHot = np.zeros(24 * 2 * 6 + 4*6)
    for i in range(0,5):
        oneHot[i*24+np.where(board[1:25] == i)[0]-1] = 1
    oneHot[5*24+np.where(board[1:25] >= 5)[0]-1] = 1
    for i in range(0,5):
        oneHot[6*24+i*24+np.where(board[1:25] == -i)[0]-1] = 1
    oneHot[6*24+5*24+np.where(board[1:25] <= -5)[0]-1] = 1
    
    numJail=np.abs(board[25])
    if numJail>5:
        numJail=5
    oneHot[12 * 24 + numJail]=1
    numJail=np.abs(board[26])
    if numJail>5:
        numJail=5
    oneHot[12 * 24 + 6+numJail]=1
    
    numOff=np.abs(board[27])
    if numOff>5:
        numOff=5
    oneHot[12 * 24 + 12+numOff]=1
    numOff=np.abs(board[28])
    if numOff>5:
        numOff=5
    oneHot[12 * 24 + 18+numOff]=1
    
    return oneHot



#actor = pickle.load(open('saved_agent2', 'rb'))
actor=agent()
numGames=10001
learnit(numGames, actor)
numWins=[]


#with open('test_pickle.pkl', 'wb') as pickle_out:  
#    pickle.dump(agent, pickle_out)
#    
#file_net = open('saved_greedy', 'wb')
#pickle.dump(actor, file_net)
#file_net.close()




