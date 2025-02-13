import numpy as np
import matplotlib.pyplot as plt

class LOLA:
    def __init__(self, real_CTR, predicted_CTR, epsilon, delta, gamma, c_Phi, MODE, initial_pulls=1, init_with_pred=False, max_pulls=None, verbose=False, random_pulling=False, compare_pred=False, a_b_test=False):
        # LOLA algorithm use predicted CTR as a initialization for the mean of each arm
        # initial_pulls represents how much we trust the predicted CTR
        self.real_CTR = real_CTR
        self.predicted_CTR = predicted_CTR
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.c_Phi = c_Phi
        self.MODE = MODE # add for 'additive', multi for 'multiplicative'

        self.n_arm = len(self.real_CTR)
        self.initial_pulls = initial_pulls
        self.init_with_pred = init_with_pred
        self.max_pulls = max_pulls
        if MODE == 'add':
            self.oracle_selected_arm = [i for i in range(self.n_arm) if (self.real_CTR[i] >= max(self.real_CTR) - self.epsilon - self.gamma)]
            # we use epsilon+gamma, set epsilon to zero.
            # the algorithm will select all arms with CTR >= max(CTR) - epsilon, and reject arms with CTR < max(CTR) - epsilon - gamma
            # for arms with CTR in between, they could be either selected or rejected.
            # success is defined as the selected arms are subset of oracle_selected_arm
            # thus algorithm is selected once they found at least one arm with CTR >= max(CTR) - epsilon - gamma
        elif MODE == 'multi':
            self.oracle_selected_arm = [i for i in range(self.n_arm) if self.real_CTR[i] >= max(self.real_CTR) * (1 - self.epsilon - self.gamma)]

        self.verbose = verbose
        self.random_pulling = random_pulling
        self.compare_pred = compare_pred
        self.a_b_test = a_b_test
        # self.stop_criteria = stop_criteria

    def pull_arm(self, idx):
        if self.verbose:
            print('pull arm {}'.format(idx))
        self.n_pulling += 1
        reward = np.random.binomial(1, self.real_CTR[idx])
        if np.isnan(self.hat_mu_i[idx]):
            self.hat_mu_i[idx] = reward
            assert self.T_i[idx] == 0
        else:
            self.hat_mu_i[idx] = (self.hat_mu_i[idx] * self.T_i[idx] + reward) / (self.T_i[idx] + 1)
        self.T_i[idx] += 1  
        return reward
    
    def C(self, delta, t):
        return np.sqrt(self.c_Phi * np.log(np.log2(2 * t) / delta) / t )
    
    def update_L_t_U_t_hat_G_K(self):
        # Line 2
        max_hat_mu = np.max(self.hat_mu_i)
        if self.MODE == 'add':
            hat_G = [i for i in range(self.n_arm) if self.hat_mu_i[i] >= max_hat_mu - self.epsilon]
        elif self.MODE == 'multi':
            hat_G = [i for i in range(self.n_arm) if self.hat_mu_i[i] >= max_hat_mu * (1 - self.epsilon)]

        if self.MODE == 'add':
            # Line 3
            _ = [self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) - self.epsilon - self.gamma for i in range(self.n_arm)]
            U_t = np.max(_)

            _ = [self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) - self.epsilon for i in range(self.n_arm)]
            L_t = np.max(_)
        elif self.MODE == 'multi':
            # Line 4
            _ = [self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) for i in range(self.n_arm)]
            U_t = (1 - self.epsilon - self.gamma) * np.max(_)

            _ = [self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) for i in range(self.n_arm)]
            L_t = (1 - self.epsilon) * np.max(_)

        val_1 = np.min([self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) - L_t for i in range(self.n_arm)])
        val_2 = np.max([self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) - U_t for i in range(self.n_arm)])

        # Line 5
        K = [i for i in range(self.n_arm) if (self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) < L_t or self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) > U_t)]

        info = {'val_1': val_1, 'val_2': val_2}

        return hat_G, K, info


    def run(self):
        if self.a_b_test:
            return self.run_a_b()
        else:
            return self.run_lola()
        
    def run_a_b(self):
        self.n_pulling = 0
        self.hat_mu_i = np.ones(self.n_arm) * np.nan
        self.T_i = np.zeros(self.n_arm)
        while self.n_pulling < self.max_pulls * self.n_arm:
            i = np.random.randint(self.n_arm)
            self.pull_arm(i)
        
        # selected_arm = [np.argmax(self.hat_mu_i)]
        # U_t = np.max([self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) for i in range(self.n_arm)])
        # selected_arm = [i for i in range(self.n_arm) if self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) > U_t]
        # if len(selected_arm) == 0:
        selected_arm = [np.argmax(self.hat_mu_i)]
        return selected_arm, self.oracle_selected_arm, self.n_arm, self.n_pulling, self.check_success(selected_arm)
        
    def run_lola(self):
        # run the LOLA algorithm, report the number of total pulling needed
        self.n_pulling = 0
        if self.compare_pred:
            # use predicted CTR directly, return arms with predicted CTR >= max(predicted CTR) - epsilon
            selected_arm = [i for i in range(self.n_arm) if self.predicted_CTR[i] >= max(self.predicted_CTR) - self.epsilon - self.gamma]
        else:
            # Line 1, but LOLA version
            if self.init_with_pred:
                self.T_i = np.ones(self.n_arm) * self.initial_pulls
                self.hat_mu_i = self.predicted_CTR.copy()
            else:
                self.hat_mu_i = np.ones(self.n_arm) * np.nan
                self.T_i = np.zeros(self.n_arm)
                # self.T_i = np.ones(self.n_arm) # T_i: the number of times arm i has been pulled
                [self.pull_arm(i) for i in range(self.n_arm)] # hat_mu_i: the empirical mean of arm i

            hat_G, K, info = self.update_L_t_U_t_hat_G_K()
            val_1_ls, val_2_ls = [], []
            hat_mu_rec = []
            T_i_rec = []
            val_1_ls.append(info['val_1'])
            val_2_ls.append(info['val_2'])

            def visualize():
                plt.plot(np.log(np.array(val_1_ls)), label='val_1')
                plt.plot(-np.log(-np.array(val_2_ls)), label='val_2')
                plt.legend()
                plt.savefig('tmp')
                plt.close()


                plt.figure(figsize=(10,10))
                plt.subplot(211)
                if len(hat_mu_rec) > 100:
                    for i in range(self.n_arm):
                        plt.plot([_[i] for _ in hat_mu_rec[100:]], label=str(i), c='C{}'.format(i))
                        plt.hlines(y=self.real_CTR[i], xmin=0, xmax=n_period, color='C{}'.format(i), linestyle='--')
                plt.hlines(y=max(self.real_CTR) - self.epsilon, xmin=0, xmax=n_period, color='black', linestyle='dotted', label='max - epsilon', alpha=0.5)
                plt.hlines(y=max(self.real_CTR) - self.epsilon - self.gamma, xmin=0, xmax=n_period, color='black', linestyle='dotted', label='max - epsilon - gamma', alpha=0.5)
                plt.legend()
                plt.title('hat_mu')

                plt.subplot(212)
                for i in range(self.n_arm):
                    plt.plot([_[i] for _ in T_i_rec], label=str(i))
                plt.legend()
                plt.title('T_i')
                plt.tight_layout()
                plt.savefig('tmpp')
                plt.close()


            n_period = 0
            VIS_INTERVAL = 1000

            # while True: # Line 6
            #     # # if self.stop_criteria == 'both':
            #     #     if len(K) == self.n_arm or self.n_pulling >= self.max_pulls * self.n_arm:
            #     #         break
            #     # elif self.stop_criteria == 'pulls':
            #     #     if self.n_pulling >= self.max_pulls * self.n_arm:
            #     #         break
            while len(K) != self.n_arm:
                if self.max_pulls is not None and self.n_pulling >= self.max_pulls * self.n_arm:
                    break

                n_period += 1
                if self.verbose:
                    print(K)


                # Line 7
                if self.random_pulling:
                    i_1 = np.random.randint(self.n_arm)
                    self.pull_arm(i_1)
                else:
                    hat_G_minus_K = [i for i in hat_G if i not in K]
                    if len(hat_G_minus_K) == 0:
                        pass
                    else:
                        _ = {i: self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) for i in hat_G_minus_K}
                        i_1 = min(_, key=_.get)
                        self.pull_arm(i_1)

                # Line 8
                if self.random_pulling:
                    i_2 = np.random.randint(self.n_arm)
                    self.pull_arm(i_2)
                else:
                    hat_G_C_minus_K = [i for i in range(self.n_arm) if i not in K and i not in hat_G]
                    if len(hat_G_C_minus_K) == 0:
                        pass
                    else:
                        _ = {i: self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) for i in hat_G_C_minus_K}
                        i_2 = max(_, key=_.get)
                        self.pull_arm(i_2)
                
                # Line 9
                if self.random_pulling:
                    i_star = np.random.randint(self.n_arm)
                    self.pull_arm(i_star)
                else:
                    i_star = np.argmax([self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) for i in range(self.n_arm)])
                    self.pull_arm(i_star)

                # update T_i_rec and hat_mu_rec
                hat_mu_rec.append(self.hat_mu_i.copy())
                T_i_rec.append(self.T_i.copy())

                # Line 10
                hat_G, K, info = self.update_L_t_U_t_hat_G_K()
                val_1_ls.append(info['val_1'])
                val_2_ls.append(info['val_2'])

                if n_period % VIS_INTERVAL == 0:
                    if self.verbose:
                        visualize()
            if self.verbose:
                visualize()

            _ = [self.hat_mu_i[i] + self.C(self.delta/self.n_arm, self.T_i[i]) - self.epsilon - self.gamma for i in range(self.n_arm)]
            U_t = np.max(_)
            selected_arm = [i for i in range(self.n_arm) if self.hat_mu_i[i] - self.C(self.delta/self.n_arm, self.T_i[i]) > U_t]

            if len(selected_arm) == 0:
                selected_arm = [np.argmax(self.hat_mu_i)]

        return selected_arm, self.oracle_selected_arm, self.n_arm, self.n_pulling, self.check_success(selected_arm)

    def check_success(self, selected_arm):
        # if set(selected_arm) is subset of self.oracle_selected_arm, return True
        # return set(selected_arm) == set(self.oracle_selected_arm)
        if len(selected_arm) == 0:
            print("##### selected arm is empty #####")
            print(self.n_pulling)
            # Ideally, if this algorithm runs for enough time, it should always find at least one arm with CTR >= max(CTR) - epsilon - gamma
            # but we truncated the running time to max_pulls, thus it may not find any arm satisfying the condition
        return set(selected_arm).issubset(set(self.oracle_selected_arm)) and len(selected_arm) >= 1