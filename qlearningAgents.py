# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        if ((state, action) in self.values.keys()) == False:
          return 0
        else:
          return self.values[(state, action)]
          # return the Q value

        # util.raiseNotDefined() # ** ORIGINAL **


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        possActions = self.getLegalActions(state)

        if len(possActions) == 0:
          return 0
        else:

          qValList = []

          for action in possActions:
            qValList.append(self.getQValue(state, action))
          
          return max(qValList)

        # util.raiseNotDefined() # ** ORIGINAL **

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        if len(self.getLegalActions(state)) == 0:
          return None

        action_vals = list()

        for action in self.getLegalActions(state):
            action_vals.append((self.getQValue(state, action), action))
        
        if len(action_vals) != len(set(action_vals)):

          maxQNum = max(action_vals)[0]
          action_valsTied = []

          for actionTuple in action_vals:
            if actionTuple[0] == maxQNum:
              action_valsTied.append(actionTuple)

          return random.choice(action_valsTied)[1]

        return max(action_vals)[1]

        # util.raiseNotDefined() # ** ORIGINAL **

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if len(legalActions) == 0:
          return None

        probability = util.flipCoin(self.epsilon)

        randAction = random.choice(legalActions)

        if probability:
          return randAction
        else:
          action_vals = list()

          for action in legalActions:
            action_vals.append( (self.values[(state, action)], action) )

          return max(action_vals)[1]

        # util.raiseNotDefined() # ** ORIGINAL **

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        sample = reward + ( self.discount * self.computeValueFromQValues(nextState) )

        quantity1 = (1 - self.alpha) * (self.values[(state, action)])
        quantity2 = (self.alpha) * sample

        # print(self.values)

        self.values[(state, action)] = quantity1 + quantity2

        # util.raiseNotDefined() # ** ORIGINAL **

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        total = 0

        for feature in self.featExtractor.getFeatures(state, action).items():
          total += self.weights[feature] * feature[1]

        return total

        # util.raiseNotDefined() # ** ORIGINAL **

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        weightsCopy = self.getWeights().copy()

        for feature in self.featExtractor.getFeatures(state, action).items():
          difference = (reward + ( self.discount * self.computeValueFromQValues(nextState) )) - self.getQValue(state, action)
          quantity1 = self.weights[feature]
          quantity2 = (self.alpha) * difference * feature[1]

          weightsCopy[feature] = quantity1 + quantity2

        for feature in self.featExtractor.getFeatures(state, action).items():
          self.weights[feature] = weightsCopy[feature]

        # util.raiseNotDefined() # ** ORIGINAL **

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass