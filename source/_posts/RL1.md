---
title: RL1 Introduction to RL
mathjax: true
date: 2017-03-30 23:18:44
categories: reinforcement learning
tags: [reinforcement learning, machine learning]
---
reference:
    [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
# lecture 1 Introduction to Reinforcement Learning
## reinforcement learning feature
1. no supervisor, only a reward signal 
1. delayed feedback
1. time matters
1. agent's actions affect the subsequent data

> **Reward Hypothesis**
> all goal can be described by the maximisation of expected cumulative reward.

## environment state and agent state
environment state : whatever data to environment
agent state: whatever data to agent
> **information state**
> Information state is Markov

## Fully Observable Environments and Partially Observable Environments

## Major Components of an RL Agent
1. Policy
1. Value function
1. Model
    1. A model predict what the environment will do next
    1. P predict the next state
    1. R predict the next reward

> **Maze Example**
> Agent may have an internal model of the environment 
> Dynamics: how actions change the state
> Rewards: how much reward from each state
> The model may be imperfect

## Categorize RL agents
* value based
* policy based
* Actor Critic

* Model Free
* Model Based

## Learning and Planning 
Two fundamental problems in sequential decision making
1. Reinforcement learning 
    * The environment is initially unknown
    * The agent interacts with the environment 
    * The agent improves its policy
1. Planning
    * A model of the environment is known
    * The agent performs computation with its model
    * The agent improves its policy

## Exploration and Exploitation
1. Reinforcement learning is like trail-and-error learning
1. the agent should discover a good policy
1. From its experience of the environment 
1. Without losing too much reward along the way

## Prediction and Control
1. Prediction: evaluate the future, **given a policy**
1. Control: optimize the future, **find the best policy**
