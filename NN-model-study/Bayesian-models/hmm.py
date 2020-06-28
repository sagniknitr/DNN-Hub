import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from matplotlib import pylab as plt
import scipy.stats

class HMM : 
    def __init__(self, num_states, x, rates, durations) :
        self.num_states = 4
        #self.true_rates = [40, 3, 20, 50]
        #self.durations = [10, 20, 5, 35]
        self.observed_counts = np.concatenate([
        scipy.stats.poisson(_rate).rvs(_num_steps)
        for (_rate, _num_steps) in zip(rates, durations)
        ]).astype(np.float32)

    def transition_matrix(self) :
        _initial_state_logits = np.zeros([self.num_states], dtype=np.float32) # uniform distribution

        _daily_change_prob = 0.05
        self._transition_probs = daily_change_prob / (num_states-1) * np.ones(
        [self.num_states, self.num_states], dtype=np.float32)
        np.fill_diagonal(_transition_probs,
                 1-_daily_change_prob)
        
        print("Initial state logits:\n{}".format(initial_state_logits))
        print("Transition matrix:\n{}".format(transition_probs))
    
    def train_HMM(self) :
        # Define variable to represent the unknown log rates.
        _trainable_log_rates = tf.Variable(
        np.log(np.mean(self.observed_counts)) + tf.random.normal([self.num_states]),
        name='log_rates')

        self.hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(
            logits=initial_state_logits),
        transition_distribution=tfd.Categorical(probs=self._transition_probs),
        observation_distribution=tfd.Poisson(log_rate=_trainable_log_rates),
        num_steps=len(observed_counts))








true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]

plt.plot(observed_counts)
#plt.show()


rate_prior = tfd.LogNormal(5, 5)

def log_prob():
 return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
         hmm.log_prob(observed_counts))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

@tf.function(autograph=False)
def train_op():
  with tf.GradientTape() as tape:
    neg_log_prob = -log_prob()
  grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
  optimizer.apply_gradients([(grads, trainable_log_rates)])
  return neg_log_prob, tf.math.exp(trainable_log_rates)

for step in range(201):
  loss, rates = [t.numpy() for t in train_op()]
  if step % 20 == 0:
    print("step {}: log prob {} rates {}".format(step, -loss, rates))

print("Inferred rates: {}".format(rates))
print("True rates: {}".format(true_rates))


# Runs forward-backward algorithm to compute marginal posteriors.
posterior_dists = hmm.posterior_marginals(observed_counts)
posterior_probs = posterior_dists.probs_parameter().numpy()

def plot_state_posterior(ax, state_posterior_probs, title):
  ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | counts)')
  ax.set_ylim(0., 1.1)
  ax.set_ylabel('posterior probability')
  ax2 = ax.twinx()
  ln2 = ax2.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
  ax2.set_title(title)
  ax2.set_xlabel("time")
  lns = ln1+ln2
  labs = [l.get_label() for l in lns]
  ax.legend(lns, labs, loc=4)
  ax.grid(True, color='white')
  ax2.grid(False)

fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(2, 2, 1),
                     posterior_probs[:, 0],
                     title="state 0 (rate {:.2f})".format(rates[0]))
plot_state_posterior(fig.add_subplot(2, 2, 2),
                     posterior_probs[:, 1],
                     title="state 1 (rate {:.2f})".format(rates[1]))
plot_state_posterior(fig.add_subplot(2, 2, 3),
                     posterior_probs[:, 2],
                     title="state 2 (rate {:.2f})".format(rates[2]))
plot_state_posterior(fig.add_subplot(2, 2, 4),
                     posterior_probs[:, 3],
                     title="state 3 (rate {:.2f})".format(rates[3]))
plt.tight_layout()
plt.show()