Marketing Analytics - Homework 2
A/B Testing with Multi-Armed Bandits

In this homework, I implemented Epsilon-Greedy and Thompson Sampling to compare four advertisement options. The goal was to see how the algorithms learn to select the advertisement with the highest average reward.
The experiment uses four bandits with mean rewards of 1, 2, 3 and 4 and each algorithm runs for 20,000 trials.
I used Gaussian rewards with a known precision of 1. For Epsilon-Greedy, epsilon decreases as 1/t. For Thompson Sampling, I used a Gaussian posterior to update the estimated rewards.
The program generates graphs comparing the learning process, cumulative rewards and cumulative regrets and saves the experiment results in a CSV file.

Results and Interpretation:

Both algorithms performed well and learned to select the advertisement with the highest mean reward of 4.
Epsilon-Greedy achieved a cumulative reward of 79860.53 and a cumulative regret of 15, while Thompson Sampling achieved a cumulative reward of 79910.77 and a cumulative regret of 2.
We can see from the learning graphs that both algorithms had some fluctuations at the beginning, but their average rewards gradually approached 4. Also the cumulative reward graph shows that both algorithms collected almost the same amount of reward but Thompson Sampling had a slightly higher cumulative reward and lower regret in this experiment.
Overall, Thompson Sampling performed better in this simulation, but we cannot conclude that it will always perform better because in this homework I used only one random seed.

Bonus - Better Implementation Plan:
In my opinion the experiment could be improved by following these steps:

1) First, I would separate the bandit setup, experiments, and visualizations into different functions to make the code easier to understand and update.
2) I would use different random seeds to repeat experiments instead of running each algorithm just once which would help check whether the results are consistent.
3) I would also test different epsilon and precision values to see how they affect the learning process and the final rewards.
4) Also, I would compare the average rewards and regrets across all runs and create graphs to show the results more clearly.

I think these changes would make the comparison more reliable and help us better understand the performance of both algorithms.

How to run the code:

Install the required packages:
`pip install -r requirements.txt`

Run the experiment:
`python Bandit.py`

The program generates `bandit_rewards.csv`, `learning_process.png` and `comparison.png`.