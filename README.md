# Why would you do that?
Repository for the BCS-IITK semester project "Why would you do that?" started in June 2021.

# Abstract
Neuroeconomics seeks to explain human decision making, the ability of an agent to process multiple alternatives and to follow a course of action. In general, a population of people thrives when they depict certain social behaviors. Small individual choices can have huge impacts on the population. In this project we’ll be taking a look at how multi-agent systems interact and produce macro-effects as a result of micro-choices, and how those results in turn affect the future decisions of the agents. We’ll create a simple agent that can decide whether to show altruistic traits or not, in an artificial environment to maximize it’s chances of survival (we will be creating and customizing this environment as well), we’ll simulate this on a number of populations with various constraints to find patterns. Next, we’ll try to create a simple Q-Learning model/any other suitable model to take into account the variability in the agent decisions based on environmental input (reinforcement).

# Project Timeline

#### WEEK 1 : Read about reciprocal altruism and ant-colony-optimization algorithm.
#### WEEK 2 : Read about the evolutionary game theory; learnt how to work with Pytorch.
#### WEEK 3 : Designed a model where agents evolve based on certain rules pertaining to reciprocal altruism.
#### WEEK 4 : Encoded this model on Python [Part A].
#### WEEK 5 : In-depth reading on deep learning and learnt how to work on neural networks in Python.
#### WEEK 6 : Read papers on RL, deep RL and multi-agent RL.
#### WEEK 7 : Involved a QL-tweak to our original Python implementation [Part B]

# Background

Rational decision-making in a multi-agent environment where kin-based altruism does not exist is governed by inferences drawn from the game theory, the social theory as well as conflict theories. Thus, we can say that if we understand these theories and strategies well, the decisions made by the agents can be predicted. Through this project, we are trying to understand the various rational decisions that can be made by an agent in a multi-agent environment by playing with the various modelling parameters. To do this, we set up an environment on Python with multiple agents suspended within it. We have completed this project in two parts (Part A and Part B) , keeping the same environment but changing the decision-making parameters, and each part provides us with thoughtful observations.

# The Environment

We have an environment where multiple agents of various sizes are suspended within. One iteration in our simulation refers to a single pair of day-time followed by night-time. The survival of these agents depends upon the food that they can get every day, while evolution is a secondary priority- they can only reproduce if they have a sufficient amount of food, and even this happens based on a fixed probability. 

During the day-time, an agent’s task is to collect as much food as it could. At night, the agents with excess food can choose (depending on their sharing strategy) to share their food with the needy, food-less agents. Only the agents with the larger quantity of food are given the ability to reproduce. If the needy food-less agents are unable to get food even at night, they die. It should be noted that the agents can only ask for food from other agents when they have no food for themselves. 

When it comes to sharing their excess food, our agents can adopt one of the four strategies, depending on the model chosen-
1. Always cooperative (AC) : Agents following this strategy are assigned the maximum probability to share food and get food in return. 
2. Tit-for-tat (TFT) : Agents decide whether they will share their food or not based on the history of the agent-at-mercy (needy agent).
3. Alternatively cooperate (ALT) : Sharing alternates between cooperativeness and competitiveness for every iteration of gathering food.
4. Always defective (AD) : Agents following this sharing strategy are assigned the least probability to share the food they have acquired.

# Python implementation of the environment

Our aim from the Python implementation of this multi-agent environment is to find out which strategy (or strategies) will yield the highest population by the end of our iterations (one iteration is nothing but a pair of consecutive day and night: hence it is an indicator of the time passed since the simulation began). Another thing to observe is to decide which agent size will be having the upper hand in the evolutionary process by the end of our iterations in both models. The implementation has a very probabilistic approach:
* The agents are initialized with a few attributes like size, id and strategy. The “agent id” is unique for every individual as it reflects the total population that has existed in the environment so far.
* The environment is modelled as an (n x n) matrix. Each cell in the matrix can host either one unit of food or one agent.
* Our simulation will have a day and a night in one iteration. In one round of day and night i.e. one iteration, agents get spawned at random places in an empty grid. Similarly, grids are randomly populated with food on a daily basis.
* The agent that is close to the food gets an advantage over grabbing the food. 
* No limit is set over the quantity of food that an individual could acquire. This is done deliberately to observe the sharing strategies that the agents will adopt at the end of the day.
* In cases of conflict, we refer to the priority of directions, which we accomplished by using the breadth-first search (BFS) algorithm. The priority order of the directions is:
 ``` 
dirn = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
 ```
 * The agents can possess any number of “food-values” starting from 0. At the end of the day, if the agent has more than 1 food-value, it gets to choose if it prefers to share it with other needy agents or keep just it for itself.
* Each agent is checked for food at night and three special cases can arise:
   * Case 1 : 0 food-value: If it has no food it is very much likely to get eliminated. It will not get eliminated if it manages to acquire food from a cooperating/sharing agent (see the next case).
   * Case 2 : Greater than 1 food-value: If the agent has more than 1 unit of food, it can decide whether it will share its food or not based on its own strategy; and its decision will also be based upon the past strategies of the agent-at-mercy (needy agent).
   * Case 3 : Greater than 2 food-values: In this case, the agent is provided with the ability to reproduce. It can decide whether it will reproduce or not based on a fixed probability provided by us for the model.

# PART A  
##  This part tries to draw conclusions on what sharing strategy thrives the best in our environment

In this part of the project, all agents are following a fixed sharing policy throughout their lifespan. However, to understand the evolution of the agents better, we decided to split our simulation into two different models:

1. **Model 1 :** Each agent is allowed to use any one of the four sharing strategies for its entire lifetime. There will be a low (but fixed) chance of reproduction for agents. 
2. **Model 2 :** The TFT strategy cannot be used by the agents in this model. Thus, they can use only one of the three strategies for their entire lifespan. There will be a (comparatively) higher (and fixed) chance of reproduction for the agents in this simulation. Food quantity is less than that available for Model 1.

Another notable decision-making parameter added in this part of the project is the size factor. For this, we are assuming that the agents are of two sizes: Size 1 and Size 2, such that Size 2 > Size 1. In both of these models, the agents of Size 2 are going to be selectively altruistic (more willing to share their food) towards needy agents of Size 1. Mathematically, this implies that the probability of an agent of Size 2 to share its food increases when it meets an agent of Size 1 than when it meets an agent of Size 2.

### Python implementation 
The two models have different, but fixed probabilities for reproduction in Part A. This means that every agent in both models has a fixed chance to reproduce. Following code determines the probability of sharing by an agent having more than 1 unit of food:
```
prob = (agent.size/(atmercy.size+agent.size))*agent.strat[0]*(atmercy.strat[-1] if atmercy.strat != [] else 1)
```
After implementing the above mentioned on Python and reiterating the simulation on the 1st model 44 times while a bit lesser (40 times) on the 2nd model, we find ourselves being able to analyze the best strategies for both the models after the iterations. We are also able to figure out which size will be more favoured for survival in the long run for both models *(the number of iterations passed simultaneously represents the time that has passed in this particular environment.)*














