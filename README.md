# DynamicObstacleAvoidance

## Motivation
For Self Driving Cars to function we need to know the current and future state (positions and velocities) of
the agents (pedestrians, other cars, etc) in the environment. This information is important for us to find
the best path for our car to travel on. In our project we want to predict the motion of these agents given
their temporal data. This motion prediction of agents can then be used to find the optimal path for our self
driving car.

## Related Work
RNNs and LSTMs have shown to be applicable to time-series forecasting. They have been successfully
applied in domain of trajectory prediction.
We will specifically use the social LSTM that can jointly predict agent trajectories by taking social interaction into account. Conceptually closest to our work is where a valid predictor is constructed using conformal prediction, and then utilized to design a controller.
Waymo has a open source dataset which can be used for this very purpose. We will also try to prepare
our own dataset and train on that using CARLA simulator.

## Problem Formulation
We plan on making a Deep Learning model, collect data from CARLA simulator and test our model on
the simulator. To predict the trajectories of agents, we will use LSTM model. We will take, say 40, past
states of agents, and predict next, say 10, states of each agent. We will also use Conformal prediction on
the LSTM model that will give us a security constraint and help us find a secure result for our self driving
car. This will give us a circular area around the predicted agent states based on the uncertainty. We will
use this circular area as a safety measure while planning for our self driving car.

## Evaluation
We will compare the predicted motion plan of multiple agents based on past temporal data. Applying
conformal prediction over the results, the final predicted trajectory will be compared to see if the output
results are within safety limits.