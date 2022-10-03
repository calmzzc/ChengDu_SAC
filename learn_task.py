import sys, os
import numpy as np
import warnings
import datetime
import time
import torch
import random

from train_model import Train
from agent import SAC
from utils import save_results, make_dir
from plot import plot_rewards_cn, plot_speed, evalplot_speed, plot_trainep_speed, plot_evalep_speed, \
    plot_power_cn, plot_unsafecounts_cn
from line import Section, Section2
from StateNode import StateNode
from MctsStateNode import MctsStateNode

# warnings.filterwarnings("ignore")

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class SACConfig:
    def __init__(self) -> None:
        self.algo = 'ChengDu_SAC'
        self.env = 'Section8'
        self.result_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/models/'  # path to save models
        self.train_eps = 500
        self.train_steps = 500
        self.eval_eps = 30
        self.eval_steps = 500
        self.gamma = 0.99
        self.mean_lambda = 1e-3
        self.std_lambda = 1e-3
        self.z_lambda = 0.0
        self.soft_tau = 1e-2
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg, seed=1):
    line = Section[cfg.env]
    # 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    state_dim = 2
    action_dim = 1
    agent = SAC(state_dim, action_dim, cfg)
    train_model = Train()
    return line, agent, train_model


def train(cfg, line, agent, train_model):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    train_flag = 1
    # ou_noise = OUNoise(1)  # 动作噪声,这里的1是动作的维度
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    unsafe_counts = []  # 记录超速次数
    ma_unsafe_counts = []  # 记录滑动平均次数
    total_t_list = []  # 全部时间列表
    total_v_list = []  # 全部速度列表
    total_a_list = []  # 全部动作列表
    total_acc_list = []  # 全部加速度列表
    total_ep_list = []  # 全部幕数列表
    total_power_list = []  # 总净能耗列表（牵引-再生）
    ma_total_power_list = []  # 滑动净能耗列表
    total_t_power_list = []  # 总牵引能耗列表
    total_re_power_list = []  # 总再生能耗列表
    node_list = []  # 节点列表
    for i_ep in range(cfg.train_eps):
        total_ep_list.append(i_ep)
        state = np.zeros(2)
        state[0] = np.array(0).reshape(1)
        state[1] = np.array(0).reshape(1)
        # ou_noise.reset()
        done = False
        ep_reward = 0
        ep_unsafe_counts = 0  # 每一幕的不安全动作次数
        i_step = 1
        limit_list = []
        A_limit_list = []
        t_list = [0]
        v_list = [0]
        a_list = [np.array(0).reshape(1)]
        acc_list = [np.array(0).reshape(1)]
        total_power = 0
        t_power = 0
        re_power = 0
        state_node = StateNode(state, 0, line, agent, i_ep, train_flag, train_model)
        # Mcts要用下面这个
        # state_node = MctsStateNode(state, 0, line, agent, i_ep, train_flag, train_model, parent=None)
        node_list.append(state_node)
        while True:
            i_step += 1
            state_node.get_last_node(node_list)
            # state_node.state_transition() # 一般动作转移
            state_node.safe_state_transition()  # Shield动作转移
            # state_node.Mcts_State_Transition()  # Shield Mcts动作转移
            total_power = total_power + state_node.t_power + state_node.re_power
            t_power += state_node.t_power
            re_power += state_node.re_power
            done, ep_unsafe_counts = state_node.get_reward(ep_unsafe_counts, total_power)  # 计算奖励
            ep_reward += state_node.current_reward
            t_list.append(state_node.next_state[0].copy())
            v_list.append(state_node.next_state[1].copy())
            a_list.append(state_node.action.copy())
            acc_list.append(state_node.acc.copy())
            limit_list.append(state_node.c_limit_speed / 3.6)
            A_limit_list.append(state_node.a_limit_speed / 3.6)

            # Memory_Buffer存储
            agent.memory.push(state_node.state.copy(), state_node.action.copy(), state_node.current_reward.copy(),
                              state_node.next_state.copy(), done)

            # 更新神经网络
            if i_step % 5 == 0:
                agent.update()

            if done:
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                total_acc_list.append(acc_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                acc_list.clear()
                limit_list.append(0)
                A_limit_list.append(0)
                break

            # 生成下一个新的节点
            state_node = StateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model)
            # Mcts要用下面这个
            # state_node = MctsStateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model, parent=None)
            node_list.append(state_node)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}，奖励：{}, 能耗  {}, 牵引能耗  {}, 最终时间  {}, 最终速度  {}, 不安全次数  {}, 最终位置 {}'.format(i_ep + 1,
                                                                                                    cfg.train_eps,
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
                                                                                                             0), state_node.step))
        rewards.append(ep_reward)
        unsafe_counts.append(ep_unsafe_counts)
        if ma_unsafe_counts:
            ma_unsafe_counts.append(0.9 * ma_unsafe_counts[-1] + 0.1 * ep_unsafe_counts)
        else:
            ma_unsafe_counts.append(ep_unsafe_counts)
        total_power_list.append(total_power)
        total_t_power_list.append(t_power)
        total_re_power_list.append(re_power)
        if ma_total_power_list:
            ma_total_power_list.append(0.9 * ma_total_power_list[-1] + 0.1 * total_power)
        else:
            ma_total_power_list.append(total_power)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list, total_ep_list, total_power_list, ma_total_power_list, unsafe_counts, ma_unsafe_counts, total_acc_list, total_t_power_list, total_re_power_list, limit_list, A_limit_list


def eval(cfg, line, agent, train_model):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    train_flag = 0
    # ou_noise = OUNoise(1)  # 动作噪声,这里的1是动作的维度
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    unsafe_counts = []  # 记录超速次数
    ma_unsafe_counts = []  # 记录滑动平均次数
    total_t_list = []  # 全部时间列表
    total_v_list = []  # 全部速度列表
    total_a_list = []  # 全部动作列表
    total_acc_list = []  # 全部加速度列表
    total_ep_list = []  # 全部幕数列表
    # total_power_list = []  # 总净能耗列表（牵引-再生）
    # ma_total_power_list = []  # 滑动净能耗列表
    # total_t_power_list = []  # 总牵引能耗列表
    # total_re_power_list = []  # 总再生能耗列表
    node_list = []  # 节点列表
    for i_ep in range(cfg.eval_eps):
        total_ep_list.append(i_ep)
        state = np.zeros(2)
        state[0] = np.array(0).reshape(1)
        state[1] = np.array(0).reshape(1)
        # ou_noise.reset()
        done = False
        ep_reward = 0
        ep_unsafe_counts = 0  # 每一幕的不安全动作次数
        i_step = 1
        t_list = [0]
        v_list = [0]
        a_list = [np.array(0).reshape(1)]
        acc_list = [np.array(0).reshape(1)]
        total_power = 0
        t_power = 0
        re_power = 0
        state_node = StateNode(state, 0, line, agent, i_ep, train_flag, train_model)
        # Mcts要用下面这个
        # state_node = MctsStateNode(state, 0, line, agent, i_ep, train_flag, train_model, parent=None)
        node_list.append(state_node)
        while True:
            i_step += 1
            state_node.get_last_node(node_list)
            # state_node.state_transition() # 一般动作转移,使用一般动作转移时需要修改奖励函数中的超速检测
            state_node.safe_state_transition()  # Shield动作转移
            # state_node.Mcts_State_Transition()  # Shield Mcts动作转移
            total_power = total_power + state_node.t_power + state_node.re_power
            t_power += state_node.t_power
            re_power += state_node.re_power
            done, ep_unsafe_counts = state_node.get_reward(ep_unsafe_counts, total_power)  # 计算奖励
            ep_reward += state_node.current_reward
            t_list.append(state_node.next_state[0].copy())
            v_list.append(state_node.next_state[1].copy())
            a_list.append(state_node.action.copy())
            acc_list.append(state_node.acc.copy())
            if done:
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                total_acc_list.append(acc_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                acc_list.clear()
                break
            # 生成下一个新的节点
            state_node = StateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model)
            # Mcts要用下面这个
            # state_node = MctsStateNode(state_node.next_state, i_step, line, agent, i_ep, train_flag, train_model, parent=None)
            node_list.append(state_node)
        print('回合：{}/{}，奖励：{}, 能耗  {}, 牵引能耗  {}, 最终时间  {}, 最终速度  {}, 不安全次数  {}, 最终位置 {}'.format(i_ep + 1,
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
                                                                                                         0), state_node.step))
        rewards.append(ep_reward)
        unsafe_counts.append(ep_unsafe_counts)
        if ma_unsafe_counts:
            ma_unsafe_counts.append(0.9 * unsafe_counts[-1] + 0.1 * ep_unsafe_counts)
        else:
            ma_unsafe_counts.append(ep_unsafe_counts)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list, total_ep_list, total_acc_list


if __name__ == "__main__":
    cfg = SACConfig()
    line, agent, train_model = env_agent_config(cfg, seed=19)
    train_time_start = time.time()
    t_rewards, t_ma_rewards, v_list, t_list, a_list, ep_list, power_list, ma_power_list, unsafe_c, ma_unsafe_c, acc_list, total_t_power_list, total_re_power_list, limit_list, A_limit_list = train(cfg, line,
                                                                                                                                                                                                    agent,
                                                                                                                                                                                                    train_model)
    train_time_end = time.time()
    train_time = train_time_end - train_time_start
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(t_rewards, t_ma_rewards, tag='train', path=cfg.result_path)
    # 测试
    line, agent, train_mdoel = env_agent_config(cfg, seed=19)
    agent.load(path=cfg.model_path)
    eval_time_start = time.time()
    rewards, ma_rewards, ev_list, et_list, ea_list, eval_ep_list, eacc_list = eval(cfg, line, agent, train_model)
    eval_time_end = time.time()
    eval_time = eval_time_end - eval_time_start
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)

    # 画图
    plot_rewards_cn(t_rewards, t_ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)  # 训练奖励
    # plot_power_cn(power_list, ma_power_list, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)  # 训练净能耗
    # plot_unsafecounts_cn(unsafe_c, ma_unsafe_c, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)  # 训练不安全次数

    plot_rewards_cn(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)  # 测试奖励

    # plot_speed(v_list, t_list, a_list, acc_list, tag="op_train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    evalplot_speed(ev_list, et_list, ea_list, eacc_list, limit_list, A_limit_list, tag="op_eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    # plot_trainep_speed(v_list, t_list, a_list, ep_list, acc_list, tag="ep_train", env=cfg.env, algo=cfg.algo,
    #                    path=cfg.result_path)
    # plot_evalep_speed(ev_list, et_list, ea_list, eval_ep_list, eacc_list, tag="ep_eval", env=cfg.env, algo=cfg.algo,
    #                   path=cfg.result_path)
    print("训练时间为{}".format(train_time))
    print("计算时间为{}".format(eval_time / 30))
