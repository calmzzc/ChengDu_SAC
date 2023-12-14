import sys, os
import numpy as np
import warnings
import datetime
import time
import torch
import random
import pandas as pd

from train_model import Train
from agent import SAC
from utils import save_results, make_dir
from plot import plot_rewards_cn, plot_speed, evalplot_speed, plot_trainep_speed, plot_evalep_speed, \
    plot_power_cn, plot_unsafecounts_cn, draw_cum_prob_curve
from line import Section, SectionS, SectionX
from StateNode import StateNode
from MctsStateNode import MctsStateNode

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class SACConfig:
    def __init__(self) -> None:
        self.algo = 'SAC_CD'
        self.env = 'Section15'
        self.train_eps = 500
        self.train_steps = 500
        self.eval_eps = 10
        self.eval_steps = 500
        self.gamma = 0.99
        self.mean_lambda = 1e-3
        self.std_lambda = 1e-3
        self.z_lambda = 0.0
        self.soft_tau = 1e-2
        self.value_lr = 1e-3
        self.soft_q_lr = 3e-4
        self.policy_lr = 1e-4
        self.shield = 1
        if self.shield:
            self.algo_n = "SSA_SAC_abl"
        else:
            self.algo_n = "no_Shield"
        self.result_path = curr_path + "/outputs/" + str(self.policy_lr) + '/' + self.algo_n + '/' + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + str(self.policy_lr) + '/' + self.algo_n + '/' + self.env + '/' + curr_time + '/models/'  # path to save models
        self.data_path = curr_path + "/outputs/" + str(self.policy_lr) + '/' + self.algo_n + '/' + self.env + '/' + curr_time + '/data/'  # path to save data
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg, seed=1):
    line = Section[cfg.env]
    # 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    state_dim = 3
    action_dim = 1
    agent = SAC(state_dim, action_dim, cfg)
    train_model = Train()
    return line, agent, train_model


def eval(cfg, line, agent, train_model):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    train_flag = 0
    # ou_noise = OUNoise(1)  # 动作噪声,这里的1是动作的维度
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    unsafe_counts = []  # 记录超速次数
    ma_unsafe_counts = []  # 记录滑动平均次数
    protect_counts = []
    ma_protect_counts = []
    total_t_list = []  # 全部时间列表
    total_v_list = []  # 全部速度列表
    total_a_list = []  # 全部动作列表
    total_acc_list = []  # 全部加速度列表
    total_ep_list = []  # 全部幕数列表
    cal_list = []
    # total_power_list = []  # 总净能耗列表（牵引-再生）
    # ma_total_power_list = []  # 滑动净能耗列表
    # total_t_power_list = []  # 总牵引能耗列表
    # total_re_power_list = []  # 总再生能耗列表
    node_list = []  # 节点列表
    total_protect_counts = 0
    for i_ep in range(cfg.eval_eps):
        total_ep_list.append(i_ep)
        state = np.zeros(3)
        state[0] = np.array(0).reshape(1)
        state[1] = np.array(0).reshape(1)
        state[2] = np.array(0).reshape(1)
        # state[3] = np.array(0).reshape(1)
        # ou_noise.reset()
        done = False
        ep_reward = 0
        ep_unsafe_counts = 0  # 每一幕的不安全动作次数
        ep_protect_counts = 0
        i_step = 1
        t_list = [0]
        v_list = [0]
        a_list = [np.array(0).reshape(1)]
        acc_list = [np.array(0).reshape(1)]
        total_power = 0
        t_power = 0
        re_power = 0
        # state_node = StateNode(state, 0, line, agent, i_ep, train_flag, train_model)
        # Mcts要用下面这个
        state_node = MctsStateNode(state, 0, line, agent, i_ep, train_flag, train_model, parent=None)
        node_list.append(state_node)
        cal_time_start = time.time()
        while True:
            i_step += 1
            state_node.get_last_node(node_list)
            if cfg.shield:
                state_node.Mcts_State_Transition_eval()  # Shield动作转移
            else:
                state_node.state_transition()  # 一般动作转移
            # state_node.state_transition() # 一般动作转移,使用一般动作转移时需要修改奖励函数中的超速检测
            # state_node.safe_state_transition()  # Shield动作转移
            # state_node.Mcts_State_Transition()  # Shield Mcts动作转移
            total_power = total_power + state_node.t_power + state_node.re_power
            t_power += state_node.t_power
            re_power += state_node.re_power
            done, ep_unsafe_counts, ep_protect_counts = state_node.get_reward(ep_unsafe_counts, total_power, ep_protect_counts)  # 计算奖励
            ep_reward += state_node.current_reward
            t_list.append(state_node.next_state[0].copy())
            v_list.append(state_node.next_state[1].copy())
            a_list.append(state_node.action.copy())
            acc_list.append(state_node.acc.copy())
            if done:
                cal_time_end = time.time()
                cal_time = cal_time_end - cal_time_start
                cal_list.append(cal_time)
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                total_acc_list.append(acc_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                acc_list.clear()
                total_protect_counts += ep_protect_counts
                break
            # 生成下一个新的节点
            # state_node = StateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model)
            # Mcts要用下面这个
            state_node = MctsStateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model, parent=None)
            node_list.append(state_node)
        print('回合：{}/{}，奖励：{}, 能耗  {}, 牵引能耗  {}, 最终时间  {}, 最终速度  {}, 不安全次数  {}, 最终位置 {}, 防护次数 {}'.format(i_ep + 1,
                                                                                                                                        cfg.eval_eps,
                                                                                                                                        np.around(ep_reward[0],
                                                                                                                                                  2),
                                                                                                                                        np.around(total_power[0],
                                                                                                                                                  4), np.around(t_power[0], 4),
                                                                                                                                        np.around(
                                                                                                                                            state_node.next_state[0],
                                                                                                                                            2), np.
                                                                                                                                        around(
                state_node.next_state[1], 2),

                                                                                                                                        np.round(ep_unsafe_counts,
                                                                                                                                                 0), state_node.step, np.round(ep_protect_counts, decimals=0)))
        rewards.append(ep_reward)
        unsafe_counts.append(ep_unsafe_counts)
        protect_counts.append(ep_protect_counts)
        if ma_unsafe_counts:
            ma_unsafe_counts.append(0.9 * unsafe_counts[-1] + 0.1 * ep_unsafe_counts)
        else:
            ma_unsafe_counts.append(ep_unsafe_counts)
        if ma_protect_counts:
            ma_protect_counts.append(0.9 * ma_protect_counts[-1] + 0.1 * ep_protect_counts)
        else:
            ma_protect_counts.append(ep_protect_counts)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('测试平均防护次数：{}'.format(total_protect_counts / cfg.eval_eps))
    print('完成测试！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list, total_ep_list, total_acc_list, cal_list


if __name__ == "__main__":
    cfg = SACConfig()
    make_dir(cfg.result_path, cfg.model_path, cfg.data_path)
    # 测试
    line, agent, train_model = env_agent_config(cfg, seed=10)
    agent.load_abl_int(path='ablactor/')
    # agent.load_int(path=cfg.model_path)
    rewards, ma_rewards, ev_list, et_list, ea_list, eval_ep_list, eacc_list, cal_list = eval(cfg, line, agent, train_model)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    # 数据导出
    output_excel = {'ev_list': ev_list[1], 'et_list': et_list[1],
                    'ea_list': ea_list[1],
                    'eacc_list': eacc_list[1]}
    output = pd.DataFrame.from_dict(output_excel, orient='index')
    output.to_excel(cfg.data_path + '{}_data.xlsx'.format(cfg.algo_n + '_' + cfg.algo), index=False)
