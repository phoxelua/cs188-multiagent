# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # print successorGameState.getPowerPellet()

        currPos, currFood = currentGameState.getPacmanPosition(), currentGameState.getFood()
        currGhostPos, newGhostPos = currentGameState.getGhostPositions(), successorGameState.getGhostPositions()

        #MEHTOD3: Use fewer factors, keep in range centered around deltascore
        # print successorGameState.getScore() , currentGameState.getScore() 
        # return successorGameState.getScore() - currentGameState.getScore()

        foodScore = 1 if len(newFood.asList())==0 else 1/float(min([10000000000] + [util.manhattanDistance(newPos, foodPos) for foodPos 
                                                    in newFood.asList()])) 

        ghostScore = 0
        closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos])
        if closestGhost < 2: ghostScore = -200

        #WTF food score is like everything
        return (successorGameState.getScore() - currentGameState.getScore()) + foodScore + ghostScore

        # METHOD2: TRY LINEAR COMB OF FACTORS: Scores 8/10 but gets really low scores - TOO COMPLEX
        # foodCoeff, powerCoeff, ghostCoeff, scareCoeff = 2.5,1,1,1     
        # foodScore, powerScore, ghostScore, scareScore = 0.0,0.0,0.0,0.0

        # #Calculate foodScore
        # if newPos in currFood.asList(): foodScore += 5
        # #concat [1] to prevent zero division and to represent when no food items left (very good)
        # closestFood = 1 / float(min([1] + [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]))


 
        # #Calculate powerPellet
        # closestScared = 0
        # if max(newScaredTimes)!=0:
        #   closestScared = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos])
        # scaredScore = sum(newScaredTimes) + closestScared


        # #Calculate ghostScore (higher better)
        # if newPos in currGhostPos or newPos in newGhostPos: ghostScore -= 10000
        # #concat [5] to prevent zero division and to represent when clostest ghost very far away
        # closestGhost = float(min([5] + [util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]))
        # ghostScore += closestGhost

        # # print newGhostPos

        # if max(newScaredTimes)!=0 : ghostCoeff *=1.5

        # return foodCoeff*foodScore + powerCoeff*powerCoeff + ghostCoeff*ghostScore + scareCoeff*scareScore

        # ghostScore = 0
        # deltafood = -len(newFood.asList()) + len(currFood.asList())
        # if newPos in currFood.asList(): deltafood += 5
        # closestFood = max(min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]),1)
        # deltafood += 1/closestFood

        # print newScaredTimes


        # currGhostDist = max(min([util.manhattanDistance(currPos, ghostPos) for ghostPos in newGhostPos]),0)
        # newGhostDist = max(min([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]),0)
        # if newGhostDist == 0 or currGhostDist == 0: 
        #     ghostScore -= 100

        # return deltafood + ghostScore
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def maxS(state,depth, agentIndex):
          depth -= 1 #THIS WAS THE PROBLEM- you must dec IMMEDIATELY 
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          v = float("-inf")
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = minS(successor,depth,agentIndex+1)[0]
            if score > v:
              v = score
              maxAction = action
          return (v,maxAction)

        def minS(state,depth,agentIndex):
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          v = float("inf")
          evalfunc, nextAgent = (minS, agentIndex+1) if agentIndex < state.getNumAgents()-1 else (maxS, 0)
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = evalfunc(successor,depth,nextAgent)[0]
            if score < v:
              v = score
              minAction = action          
          return (v,minAction)
        return maxS(gameState,self.depth,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxS(state,depth, agentIndex, alpha, beta):
          depth -= 1 #THIS WAS THE PROBLEM- you must dec IMMEDIATELY 
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          v = float("-inf")
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = minS(successor,depth,agentIndex+1, alpha, beta)[0]
            if score > v:
              v = score
              maxAction = action
            if v > beta:
              return (v,maxAction)
            alpha = max(alpha,v)
          return (v,maxAction)

        def minS(state,depth,agentIndex, alpha, beta):
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          v = float("inf")
          evalfunc, nextAgent = (minS, agentIndex+1) if agentIndex < state.getNumAgents()-1 else (maxS, 0)
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = evalfunc(successor,depth,nextAgent, alpha, beta)[0]
            if score < v:
              v = score
              minAction = action
            if v < alpha:
              return (v,minAction)
            beta = min(beta,v)          
          return (v,minAction)
        
        return maxS(gameState,self.depth,0, float("-inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        """Dispatches to minValue and maxValue based on agent index and handles search depth"""
        def value(agentIndex, depthLeft, state):
          #base case: second predicate super important (bug fixed!)
          if depthLeft == 0 and agentIndex == 0: return valAction(self.evaluationFunction(state), None) #what if depth != 0 but you win/lose?
          return maxValue(agentIndex, depthLeft-1, state) if agentIndex == 0 else expValue(agentIndex,depthLeft, state)

        """Returns a valAction object that represents the max attainable value at this node"""
        def maxValue(agentIndex, depthLeft, state):
          vA = valAction(float("-inf"), None)
          if len(state.getLegalActions(agentIndex)) == 0: return value(0,0,state)
          for action in state.getLegalActions(agentIndex):
            v = value((agentIndex+1)%state.getNumAgents(), depthLeft, state.generateSuccessor(agentIndex,action))
            if v.val > vA.val: vA = valAction(v.val, action)
          return vA

        """Returns a valAction object that represents the min attainable value at this node"""
        def expValue(agentIndex,depthLeft,state):
          n = len(state.getLegalActions(agentIndex))
          if n == 0: return value(0,0,state)
          val = sum([value((agentIndex+1)%state.getNumAgents(), depthLeft,state.generateSuccessor(agentIndex,action)).val 
                  for action in state.getLegalActions(agentIndex)]) / float(n)
          return valAction(val, None)

        return value(0, self.depth, gameState).action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghosts-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodCoeff, powerCoeff, ghostCoeff= 1,3,1,     
    foodScore, powerScore, ghostScore= 0,0,0,

    currPos, currFood = currentGameState.getPacmanPosition(), currentGameState.getFood()
    currGhostPos, currPower =  currentGameState.getGhostPositions(), currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    isScared = True if max(scaredTimes)!=0 else False

    closestFood = float(min([currFood.width + currFood.height] + [util.manhattanDistance(currPos, foodPos) for foodPos in currFood.asList()]))
    closestGhost = float(min([util.manhattanDistance(currPos, ghostPos) for ghostPos in currGhostPos]))
    closestPow = float(min([len(currPower)] + [util.manhattanDistance(powerPos, currPos) for powerPos in currPower]))


    foodScore = 1 if len(currFood.asList())==0 else 1/closestFood
    powerScore = 1 if len(currPower)==0 else 1/closestPow
    ghostScore = -100 if closestGhost < 1 else 1/closestGhost #*100 if isScared else closestGhost

    if isScared and closestGhost < max(scaredTimes): ghostCoeff, ghostScore = 100, abs(ghostScore)
    # if not isScared and closestGhost > 3: foodCoeff, ghostCoeff = 200, -1
    # if isScared:
    #   if closestGhost < max(scaredTimes):
    #     foodCoeff, powerCoeff, ghostCoeff = 1,0,1000
    # else:
    #   if closestGhost > closestPow:
    #     foodCoeff, powerCoeff, ghostCoeff = 200, 100,100
    #   elif closestGhost < closestPow:
    #     foodCoeff, powerCoeff, ghostCoeff = 200000,100,400

    return foodCoeff*foodScore + ghostCoeff*ghostScore + powerCoeff*powerScore + currentGameState.getScore()
    # return foodCoeff*foodScore + powerCoeff*powerCoeff + ghostCoeff*ghostScore + scareCoeff*scareScore

    # util.raiseNotDefined()

class valAction:

  def __init__(self, val, action):
    self.val = val
    self.action = action

# Abbreviation
better = betterEvaluationFunction

