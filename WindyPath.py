# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:06:39 2018

SARSA vs. Q-Learning approaches to the cliff walking algorithm

@author: Cameron Hargreaves
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def movePlayer(position, epsilon, estimateRewards):
    movements = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # down, up, left, right movements
    if np.random.rand() < epsilon:          # Choose a random direction in an exploratory fashion
        movementChoice = movements[np.random.randint(len(movements))]   
    else:
        movementChoice = getMaxValue(position, estimateRewards)
       
    newPosition = np.add(position, movementChoice)
    newPosition = defineBoundaries(newPosition) # Sanity check in case we have moved outside the grid
    
    return np.array(newPosition)     # Move to that square

def getMaxValue(position, estimateRewards):
    movements = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # down, up, left, right movements
    value = []
    for movement in movements:                          # See which of the four directions has the greatest reward
        newPosition = np.add(position, movement)
        if 0 <= newPosition[0] <= 3 and 0 <= newPosition[1] <= 11:
            value.append(estimateRewards[tuple(newPosition)]) 
        else:
            value.append(-200)                          # If out of bounds simply give smaller reward than others, shouldn't be called in normal operation
    
    maxValue = max(value)
    j = 0
    maxValues = []
    
    for i in value:     # get indices of the max values
        if i == maxValue:
            maxValues.append(j)
        j+=1
            
    maxIndex = random.choice(maxValues)
    return movements[maxIndex]
    
def sarsa(estimateRewards, trueRewards, start, goal):
    alpha = 0.1
    position = start
    sumReward = 0
    count = 0
    while position != goal:
        newPosition = movePlayer(position, 0.05, estimateRewards)
        
        reward = trueRewards[tuple(newPosition)]
        sumReward += reward
        
        valueTarget = estimateRewards[tuple(newPosition)]
        # update via sarsa
        estimateRewards[tuple(position)] = (estimateRewards[tuple(position)] + 
                                            alpha * 
                                            (reward + valueTarget - estimateRewards[tuple(position)]))
        count += 1
      
        position = windyPath(position)  # Move back to the start if on the cliff
        position = newPosition.tolist()
    return estimateRewards, sumReward

def qLearning(estimateRewards, trueRewards, start, goal):
    alpha = 0.1
    position = start
    sumReward = 0
    count = 0
    while position != goal:
        newPosition = movePlayer(position, 0, estimateRewards)
        reward = trueRewards[tuple(newPosition)]
        sumReward += reward
        
        valueTarget = estimateRewards[tuple(newPosition)]
        # update via Q-Learning
        estimateRewards[tuple(position)] = (estimateRewards[tuple(position)] + 
                                            alpha * 
                                            (reward + valueTarget - estimateRewards[tuple(position)]))
        count += 1
        position = windyPath(position)
        position = newPosition.tolist()
    return estimateRewards, sumReward

def defineBoundaries(position):
    if position[0] < 0:     # Stop falling off the top
        position[0] = 0
    elif position[0] > 3:    # Stop falling off the bottom
        position[0] = 3  
    if position[1] < 0:   # Stop falling off the left
        position[1] = 0        
    elif position[1] > 11:  # Stop falling off the right
        position[1] = 11              
    return position

def windyPath(position):
    if position[0] == 3 and position[1] in list(range(1,11)):  # If we are in windy path return to start
        position = [3,0]    
    return position


def movingAverage(inputArray):      # Calculate a ten point moving average of an input array
    N = 10
    cumSum, inputMovingAve = [0], []
    
    for i, x in enumerate(inputArray, 1):
        cumSum.append(cumSum[i-1] + x)
        if i>=N:
            movingAve = (cumSum[i] - cumSum[i-N])/N
            inputMovingAve.append(movingAve)
    return inputMovingAve

# Set the rewards for each of the squares
trueRewards = np.full([4, 12], -1)
trueRewards[3,1:11] = -100

estimateRewards = np.zeros([4, 12])    # Set our initial state-action reward value estimates to zero

start = [3,0]           # Starting position
goal = [3,11]           # Ending position

position = start
epsilon = 0.1

numRuns = 10                # Number of times to run the algorithm
numEpisodes = 500

sarsaEstimatesSum = np.zeros([4, 12])   # To calculate average estimate
qEstimatesSum = np.zeros([4, 12])

sarsaRewardsSum = np.zeros(numEpisodes) # To calculate average reward
qRewardsSum = np.zeros(numEpisodes)

for i in range(numRuns):    
    qEstimates = np.copy(estimateRewards)       # make copies of the reward matrix
    sarsaEstimates = np.copy(estimateRewards)
    
    qRewards = np.arange(numEpisodes)
    sarsaRewards = np.arange(numEpisodes)
    for i in range(numEpisodes):
        sarsaEstimates, reward = sarsa(sarsaEstimates, trueRewards, start, goal)
        sarsaRewards[i] = reward
    
    for i in range(numEpisodes):
        qEstimates, reward = qLearning(qEstimates, trueRewards, start, goal)
        qRewards[i] = reward
        
    sarsaEstimatesSum = np.add(sarsaEstimatesSum, sarsaEstimates)  # Add our returned estimates to our sum
    qEstimatesSum = np.add(qEstimatesSum, qEstimates)
    
    sarsaRewardsSum = np.add(sarsaRewardsSum, sarsaRewards)         # Add our returned rewards to our sum
    qRewardsSum = np.add(qRewardsSum, qRewards)
    
sarsaEstimate = np.divide(sarsaEstimatesSum, numRuns)               # Calculate the average
sarsaRewards = np.divide(sarsaRewardsSum, numRuns)
qEstimate = np.divide(qEstimatesSum, numRuns)
qRewards = np.divide(qRewardsSum, numRuns)

sarsaMoving = movingAverage(sarsaRewards)
qMoving = movingAverage(qRewards)

plt.plot(range(len(sarsaMoving)), sarsaMoving, label="Sarsa")
plt.plot(range(len(qMoving)), qMoving, label="Q-Learning")
plt.ylim(-100, 0)
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Average Reward per Episode")
plt.show