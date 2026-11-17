import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger

#Experiment settings
bandit_means = [1, 2, 3, 4]
num_trials = 20000
precision = 1


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#
class Visualization:
    """Plots the results of both algorithms."""
    def __init__(self, eg, ts):
        self.eg = eg
        self.ts = ts
        

    def plot1(self):
        """Shows the learning process on linear and log scales."""
        trials = np.arange(1, len(self.eg.rewards) + 1)

        eg_avg = np.cumsum(self.eg.rewards) / trials
        ts_avg = np.cumsum(self.ts.rewards) / trials

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        #Linear scale
        axes[0].plot(trials, eg_avg, label="Epsilon-Greedy")
        axes[0].plot(trials, ts_avg, label="Thompson Sampling")
        axes[0].axhline(max(bandit_means), color="gray",
                        linestyle="--", label="Best mean")
        axes[0].set_title("Learning Process - Linear Scale")
        axes[0].set_xlabel("Trials")
        axes[0].set_ylabel("Average Reward")
        axes[0].legend()

        #Log scale
        axes[1].plot(trials, eg_avg, label="Epsilon-Greedy")
        axes[1].plot(trials, ts_avg, label="Thompson Sampling")
        axes[1].axhline(max(bandit_means), color="gray",
                        linestyle="--", label="Best mean")
        axes[1].set_xscale("log")
        axes[1].set_title("Learning Process - Log Scale")
        axes[1].set_xlabel("Trials")
        axes[1].set_ylabel("Average Reward")
        axes[1].legend()

        #Saving the learning graph
        plt.tight_layout()
        fig.savefig(Path(__file__).resolve().parent / "learning_process.png")

        if "agg" in plt.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()


    def plot2(self):
        """Compares cumulative rewards and regrets."""
        eg_rewards = np.cumsum(self.eg.rewards)
        ts_rewards = np.cumsum(self.ts.rewards)

        #Calculating cumulative regrets
        eg_regrets = np.cumsum(self.eg.regrets)
        ts_regrets = np.cumsum(self.ts.regrets)

        trials = np.arange(1, len(self.eg.rewards) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        #Cumulative rewards
        axes[0].plot(trials, eg_rewards, label="Epsilon-Greedy")
        axes[0].plot(trials, ts_rewards, label="Thompson Sampling")
        axes[0].set_title("Cumulative Rewards")
        axes[0].set_xlabel("Trials")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend()

        #Cumulative regrets
        axes[1].plot(trials, eg_regrets, label="Epsilon-Greedy")
        axes[1].plot(trials, ts_regrets, label="Thompson Sampling")
        axes[1].set_title("Cumulative Regrets")
        axes[1].set_xlabel("Trials")
        axes[1].set_ylabel("Total Regret")
        axes[1].legend()

        #Saving the comparison graph
        plt.tight_layout()
        fig.savefig(Path(__file__).resolve().parent / "comparison.png")

        if "agg" in plt.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()


#--------------------------------------#    
class EpsilonGreedy(Bandit):
    """Chooses ads using the epsilon-greedy method."""

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.n = 0

    def __repr__(self):
        return f"EpsilonGreedy(mean={self.p})"

    def pull(self):
        #Generating a reward
        return np.random.normal(self.p, 1 / np.sqrt(precision))


    def update(self, reward):
        """Updates the estimated mean after each reward."""
        self.n += 1
        self.p_estimate += (reward - self.p_estimate) / self.n


    def experiment(self):
        """Runs the epsilon-greedy experiment for 20000 trials."""
        bandits = [EpsilonGreedy(p) for p in bandit_means]

        self.rewards = []
        self.regrets = []
        self.choices = []

        best_mean = max(bandit_means)

        for t in range(1, num_trials + 1):

            #Trying each bandit once at the beginning
            if t <= len(bandits):
                chosen = t - 1

            else:
                epsilon = 1 / t

                #Explore or exploit
                if np.random.random() < epsilon:
                    chosen = np.random.randint(len(bandits))
                else:
                    chosen = np.argmax([b.p_estimate for b in bandits])

            #Getting reward and update the estimate
            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            self.rewards.append(reward)
            self.regrets.append(best_mean - bandits[chosen].p)
            self.choices.append(chosen + 1)

        self.bandits = bandits
        return self.rewards


    def report(self):
        """Prints the reward and regret results."""
        total_reward = sum(self.rewards)
        total_regret = sum(self.regrets)

        avg_reward = total_reward / len(self.rewards)
        avg_regret = total_regret / len(self.regrets)

        #results
        logger.info(f"Cumulative reward: {total_reward:.2f}")
        logger.info(f"Cumulative regret: {total_regret:.2f}")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Average regret: {avg_regret:.4f}")



#--------------------------------------#
class ThompsonSampling(Bandit):
    """Chooses ads using Thompson Sampling."""

    def __init__(self, p):
        self.p = p
        self.m = 0
        self.lambda_ = 1
        self.n = 0

    def __repr__(self):
        return f"ThompsonSampling(mean={self.p})"

    def pull(self):
        #Generating a reward
        return np.random.normal(self.p, 1 / np.sqrt(precision))

    def sample(self):
        #Sample from the posterior
        return np.random.normal(self.m, 1 / np.sqrt(self.lambda_))


    def update(self, reward):
        """Updates the posterior using the new reward."""
        self.m = (self.lambda_ * self.m + precision * reward) / (self.lambda_ + precision)
        self.lambda_ += precision
        self.n += 1


    def experiment(self):
        """Runs Thompson Sampling for 20000 trials."""
        bandits = [ThompsonSampling(p) for p in bandit_means]

        self.rewards = []
        self.regrets = []
        self.choices = []

        best_mean = max(bandit_means)

        for t in range(num_trials):

            #Choosing the bandit with the highest sample
            samples = [b.sample() for b in bandits]
            chosen = np.argmax(samples)

            #Getting reward and update the posterior
            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            self.rewards.append(reward)
            self.regrets.append(best_mean - bandits[chosen].p)
            self.choices.append(chosen + 1)

        self.bandits = bandits
        return self.rewards

    def report(self):
        """Prints the reward and regret results."""
        total_reward = sum(self.rewards)
        total_regret = sum(self.regrets)

        avg_reward = total_reward / len(self.rewards)
        avg_regret = total_regret / len(self.regrets)

        #Results
        logger.info(f"Cumulative reward: {total_reward:.2f}")
        logger.info(f"Cumulative regret: {total_regret:.2f}")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Average regret: {avg_regret:.5f}")



def comparison():
    """Runs both algorithms and saves the results."""
    
    #Running both algorithms
    np.random.seed(42)

    eg = EpsilonGreedy(1)
    eg.experiment()

    ts = ThompsonSampling(1)
    ts.experiment()

    #Saving results to CSV
    file_path = Path(__file__).resolve().parent / "bandit_rewards.csv"

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Bandit", "Reward", "Algorithm"])

        for bandit, reward in zip(eg.choices, eg.rewards):
            writer.writerow([bandit, reward, "Epsilon-Greedy"])

        for bandit, reward in zip(ts.choices, ts.rewards):
            writer.writerow([bandit, reward, "Thompson Sampling"])

    logger.info("Results saved to bandit_rewards.csv")

    return eg, ts



if __name__ == "__main__":
    #Run both algorithms
    eg, ts = comparison()

    #Show results
    logger.info("Epsilon-Greedy results")
    eg.report()

    logger.info("Thompson Sampling results")
    ts.report()

    #Show graphs
    visual = Visualization(eg, ts)
    visual.plot1()
    visual.plot2()
