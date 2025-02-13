import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_plot_data(plot_func):
    """A wrapper that save visualization data to npy file.
    
    Args:
        plot_func (function): The function that generates the plot, and should return the path to the save data (including the file name).
                              Add this decorator to the head of your function, and the input of your functions will be automatically saved
                              at the directory specified by the return of your function.
    """
    def wrapper(*args, **kwargs):
        to_save = {'args': args, 'kwargs': kwargs}
        fig_dir_name = plot_func(*args, **kwargs)
        # fig_dir_name = fig_dir_name.replace('.', '_')
        if fig_dir_name is not None:
            np.save(fig_dir_name +'.npy', to_save)
            print('save npy file of the plot to {}'.format(fig_dir_name +'.npy'))
    return wrapper

@save_plot_data
def plot(data, max_min_CTR_gap_ls, filename):
    min_val = np.inf
    max_val = -np.inf
    for method in data:
        if method in ['ST2', 'A_B']:
            min_val = min(min_val, min(data[method]['total_pulling_rec']))
            max_val = max(max_val, max(data[method]['total_pulling_rec']))
        elif method == 'LOLA':
            for initial_pulls in data[method]:
                min_val = min(min_val, min(data[method][initial_pulls]['total_pulling_rec']))
                max_val = max(max_val, max(data[method][initial_pulls]['total_pulling_rec']))


    lola_pulling_ls = [np.mean(data['LOLA'][key]['total_pulling_rec']) for key in data['LOLA']]
    init_pull_ls = [int(key) for key in data['LOLA']]
    lola_success_ls = [data['LOLA'][key]['success_rate'] * 100 for key in data['LOLA']]


    plt.plot(lola_pulling_ls, lola_success_ls, label='LOLA', marker='o')
    for i, (x, y) in enumerate(zip(lola_pulling_ls, lola_success_ls)):
        plt.text(x, y, f'{init_pull_ls[i]}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Average Number of Pulling per Test')
    plt.ylabel('Success Rate (%)')
    plt.savefig('{}_LOLA_select_init_pull.pdf'.format(filename))
    plt.close('all')

    idx = 0
    init_pull_to_show = [300]      
    for method in data:
        if method == 'ST2':  
            idx += 1 
            plt.hist(data[method]['total_pulling_rec'], bins=40, alpha=0.5, range=(min_val, max_val), density=True, label=r'$(\mathrm{ST})^2$')
            plt.xlim(min_val - 100, max_val + 100)
            # plt.title(r'$(\mathrm{ST})^2$')
        elif method == 'LOLA':
            for initial_pulls in init_pull_to_show:
                idx += 1
                plt.hist(data[method][initial_pulls]['total_pulling_rec'], bins=40, alpha=0.5, range=(min_val, max_val), density=True, label='LOLA with Initial Pulling = {}'.format(initial_pulls))
                plt.xlim(min_val - 100, max_val + 100)
    plt.tight_layout()
    plt.legend()
    plt.savefig('{} histogram.pdf'.format(filename))
    plt.close('all')


    plt.figure(figsize=(6.4 * 2, 4.8))
    plt.subplot(121)

    selected_init_pull = 300
    a_b_pulling_ls = []
    a_b_success_ls = []
    for method in data:
        if method == 'LOLA':
            n_pulls = np.mean(data[method][selected_init_pull]['total_pulling_rec'])
            success_rate = data[method][selected_init_pull]['success_rate'] * 100
            plt.scatter(n_pulls, success_rate, label='LLM + BAI (LOLA)')

        elif method == 'LLM':
            plt.scatter(0, data[method]['success_rate'] * 100, label='Pure LLM', c='C2')
        else:
            n_pulls = np.mean(data[method]['total_pulling_rec'])
            success_rate = data[method]['success_rate'] * 100
            if method == 'ST2':
                plt.scatter(n_pulls, success_rate, label='BAI')
            if 'A_B' in method:
                a_b_pulling_ls.append(n_pulls)
                a_b_success_ls.append(success_rate)


    plt.plot(a_b_pulling_ls, a_b_success_ls, label='A/B Test Using Different Number of Pulls', marker='o', color='C3')

    plt.xlabel('Average Number of Pulls per Test')
    plt.ylabel('Success Rate (%)')
    plt.legend()


    plt.subplot(122)
    selected_init_pull = 300
    a_b_pulling_ls = []
    a_b_success_ls = []
    for method in data:
        if method == 'LOLA':
            n_pulls = np.mean(data[method][selected_init_pull]['total_pulling_rec'])
            success_rate = data[method][selected_init_pull]['success_rate'] * 100
            plt.scatter(n_pulls, success_rate, label='LLM + BAI (LOLA)')
            # add text to this scatter point
            plt.text(n_pulls+210, success_rate, f'({n_pulls:.2f}, {success_rate:.2f}%)', ha='center', va='center', fontsize=10)
            

        elif method == 'LLM':
            pass
        else:
            n_pulls = np.mean(data[method]['total_pulling_rec'])
            success_rate = data[method]['success_rate'] * 100
            if method == 'ST2':
                plt.scatter(n_pulls, success_rate, label='BAI')
                plt.text(n_pulls+210, success_rate, f'({n_pulls:.2f}, {success_rate:.2f}%)', ha='center', va='center', fontsize=10)
            if 'A_B' in method:
                a_b_pulling_ls.append(n_pulls)
                a_b_success_ls.append(success_rate)

    plt.plot(a_b_pulling_ls, a_b_success_ls, label='A/B Test Using Different Number of Pulls', marker='o', color='C3')
    plt.text(a_b_pulling_ls[1]+210, a_b_success_ls[1], f'({a_b_pulling_ls[1]:.2f}, {a_b_success_ls[1]:.2f}%)', ha='center', va='center', fontsize=10)
    
    plt.xlabel('Average Number of Pulls per Test')
    plt.ylabel('Success Rate (%)')
    plt.legend()

    # get xlim and ylim
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.subplot(121)
    rect = patches.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0], linewidth=3, edgecolor='k', facecolor='none', alpha=0.2)
    plt.gca().add_patch(rect)

    plt.tight_layout()
    plt.savefig('{} Summary.pdf'.format(filename))
    plt.close('all')



    plt.figure()
    # scatter of LOLA pulling
    plt.scatter(max_min_CTR_gap_ls, data['LOLA'][300]['total_pulling_rec'], label='LOLA', alpha=0.1)
    plt.scatter(max_min_CTR_gap_ls, data['ST2']['total_pulling_rec'], label='LOLA 200',alpha=0.1)
    plt.savefig('{}_3.png'.format(filename))


    # SAVE_PLOT_DATA is false if this script is run directly, and true if it is imported as a module
    SAVE_PLOT_DATA = (__name__ != '__main__')

    if SAVE_PLOT_DATA:
        return filename
    else:
        return None



if __name__ == '__main__':
    import numpy as np
    data_name_ls = ['tmp_results/test_c_Phi0.0015.npy']
    for data_name in data_name_ls:
        data = np.load(data_name,allow_pickle=True).item()
        # remove the last .npy from the data_name
        filename = data_name[:-4]
        plot(data['args'][0],data['args'][1], filename=filename)