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

# Model Abstract

Rational decision-making in a multi-agent environment where kin-based altruism does not exist is governed by inferences drawn from the game theory, the social theory as well as conflict theories. Thus, we can say that if we understand these theories and strategies well, the decisions made by the agents can be predicted. Through this project, we are trying to understand the various rational decisions that can be made by an agent in a multi-agent environment by playing with the various modelling parameters. To do this, we set up an environment on Python with multiple agents suspended within it. We have completed this project in two parts (Part A and Part B) , keeping the same environment but changing the decision-making parameters, and each part provides us with thoughtful observations.

# The Environment

We have an environment where multiple agents of various sizes are suspended within. One iteration in our simulation refers to a single pair of day-time followed by night-time. The survival of these agents depends upon the food that they can get every day, while evolution is a secondary priority- they can only reproduce if they have a sufficient amount of food, and even this happens based on a fixed probability. 

During the day-time, an agent’s task is to collect as much food as it could. At night, the agents with excess food can choose (depending on their sharing strategy) to share their food with the needy, food-less agents. Only the agents with the larger quantity of food are given the ability to reproduce. If the needy food-less agents are unable to get food even at night, they die. It should be noted that the agents can only ask for food from other agents when they have no food for themselves. 



