import numpy as np
import torch
import random

class MctsStateNode:
    def __init__(self, state, step, line, agent, episode, train_flag, train_model, parent=None):
        self.line = line  # 区间
        self.train_model = train_model  # 列车模型
        self.agent = agent  # 神经网络
        # self.ou_noise = ou_noise  # OU噪声

        self.state = state  # 当前状态 ,0是时间，1是速度
        self.acc = 0  # 当前合加速度
        self.tm_acc = 0  # 电机牵引加速度
        self.bm_acc = 0  # 电机制动加速度
        self.g_acc = 0  # 坡度加速度
        self.c_acc = 0  # 曲率加速度
        self.action = np.array(0).reshape(1)  # 当前牵引制动力请求百分比
        self.mean = None  # 这两个变量用来描述policy_net的分布
        self.log_std = None

        self.step = step  # 当前阶段
        self.episode = episode  # 当前幕数
        self.max_step = int(self.line.length / self.line.delta_distance - 1)
        self.current_reward = 0  # 当前奖励
        self.current_q = 0  # 当前Q值

        self.last_node_state = 0  # 前一个节点的状态
        self.last_node_action = 0  # 前一个节点的动作
        self.last_node_acc = 0  # 前一个节点的加速度

        self.next_state = np.zeros(3)  # 状态转移后的状态
        self.t_power = 0.0  # 状态转移过程的牵引能耗
        self.re_power = 0  # 状态转移过程产生的再生制动能量

        self.train_flag = train_flag  # 训练测试标志位
        self.comfort_punish = 0  # 舒适度惩罚标志位
        self.speed_punish = 0  # 超速惩罚标志位

        self.current_limit_speed = 0  # 当前限速
        self.c_limit_speed = float("inf")  # 静态限速用来画图
        self.a_limit_speed = float("inf")  # ATP限速，用来画图
        self.ave_v = 0  # 本阶段平均速度
        self.p_indicator = -10  # 超速惩罚

        self.ATP_limit = float("inf")
        self.slope = 0.0

        # MCTS相关要用到的参数
        self._parent = parent
        self._selected_children = None
        self._selected_action = 0
        self._children = {}
        self._n_visits = 0
        self._s_visits = 0
        self._u = 0
        self._Q = 0
        self._R = 0
        self.next_simulate_node = None
        self.simulate_depth = 0

        self.gama = 0.9  # 衰减系数
        self.node_list = []  # 节点列表

        self.next_max_speed = 0
        self.next_max_acc = 0
        self.next_max_action = 0
        self.max_action_index = 0

        self.destroy_flag = 0  # 判断是否需要销毁该节点，初始为0，不用销毁，主要用在Mcts过程中
        self.save_flag = 0
        self.shield_flag = 0
        self.done = 0

    # # 获取最大step
    # def get_max_step(self):
    #     self.max_step = self.line.length / self.line.delta_distance

    # # MCTS需要用到的方法
    # def select(self, c_puct):
    #     self._n_visits += 1
    #     return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    #
    # def get_value(self, c_puct):
    #     self._u = c_puct * np.sqrt(np.log(self._parent._n_visits) / (self._n_visits or 1))
    #     return self._Q + self._u + self._R

    def get_simulate_depth(self):
        self.simulate_depth = 5 - self.step % 5

    def simulate_return(self):
        t = self
        while t._parent is not None:
            count = 0
            for value in self._parent._children.values():
                if value.save_flag:
                    count += 1
                    self._parent.current_reward = (self._parent.current_reward + 0.9 * value.current_reward)
            if count != 0:
                self._parent.current_reward = self._parent.current_reward / count
            else:
                self._parent.current_reward = self._parent.current_reward
            t = t._parent

    def simulate(self):
        self.get_ATP_limit()
        # self.get_next_max_speed()
        # 在simulate前已经将action传进来了
        for i in range(3):
            self.get_action()
            self.reshape_action()
            self.get_acc()
            if not self.Shield_Check():
                self.save_flag = 1
                self.get_next_state()
                self.get_power()
                self.get_simulate_reward()
                self._children[i] = MctsStateNode(self.next_state.copy(), self.step + 1, self.line, self.agent, self.episode, self.train_flag, self.train_model,
                                                  parent=self)
                self._children[i].current_reward = self.current_reward
                self._children[i].last_node_action = self.action
        if len(self._children) != 0:
            for value in self._children.values():
                if value.step % 5 != 0 and value.step < self.max_step:
                    value.simulate()
                if value.step % 5 == 0 and value.step < self.max_step:
                    value.save_flag = 1
                    value.simulate_return()
        # if len(self._children) == 0 and self.step % 5 != 0:  # 这里的删除现在是有问题的,删字典时报错
        #     t = self._parent
        #     self.save_flag = 0
        #     while t is not None:
        #         for i in range(len(t._children)):
        #             t.save_flag = t.save_flag * 0 + t._children[i].save_flag
        #         if t.save_flag:
        #             t.save_flag = 1
        #         else:
        #             t.save_flag = 0
        #         t = t._parent

    def Mcts_do(self):
        self.simulate()

    def Mcts_Start(self):
        temp_reward_dict = {}
        temp_node_dict = {}
        MctsNode = MctsStateNode(self.state, self.step, self.line, self.agent, self.episode, self.train_flag, self.train_model, parent=None)
        MctsNode.action = self.action
        MctsNode.last_node_action = self.last_node_action
        action_dict = MctsNode.expand()
        for i in range(len(MctsNode._children)):
            MctsNode._selected_children = MctsNode._children.get(i)  # 从每个根节点开始进行演绎，根节点根据自己的状态选择动作
            MctsNode._selected_children.Mcts_do()
            if len(MctsNode._selected_children._children) == 0:
                del MctsNode._children[i]
                del action_dict[i]
            else:
                temp_reward_dict[i] = MctsNode._selected_children.current_reward
                temp_node_dict[i] = MctsNode._selected_children
        if temp_reward_dict:
            temp_index = max(temp_reward_dict, key=(lambda k: temp_reward_dict[k]))
            temp_node = MctsNode._children.get(temp_index)
            # temp_action = temp_node.action  # 这里现在应该对了，把expand时候的动作存到一个dict里，通过演绎找reward最大的action，让父节点用这个action
            temp_action = action_dict.get(temp_index)
            del MctsNode
            return temp_action
        else:
            del MctsNode
            return None

    def Mcts_Check(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        next_limit_speed = self.line.speed_limit[key]
        temp_velocity = np.array(self.state[1]).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        if velocity * 3.6 > min(next_limit_speed - 2, self.ATP_limit):
            return 1
        else:
            return 0

    def get_next_max_speed(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        self.next_max_speed = min(self.ATP_limit, self.line.speed_limit[key] - 2)

    def get_max_acc_index(self):
        temp_velocity = np.array(self.state[1]).reshape(1)
        if temp_velocity > (self.next_max_speed / 3.6):
            self.next_max_acc = -np.sqrt(-self.next_max_speed / 3.6 * self.next_max_speed / 3.6 + temp_velocity * temp_velocity) / (2 * self.line.delta_distance)
        else:
            self.next_max_acc = np.sqrt(self.next_max_speed / 3.6 * self.next_max_speed / 3.6 - temp_velocity * temp_velocity) / (2 * self.line.delta_distance)
        temp_acc = self.next_max_acc - self.g_acc - self.c_acc
        if temp_acc >= 0:
            self.next_max_action = temp_acc * self.train_model.weight / self.train_model.max_traction_force
        else:
            self.next_max_action = temp_acc * self.train_model.weight / self.train_model.max_brake_force
        if self.next_max_action > 1:
            self.next_max_action = np.array(1).reshape(1)
        if self.next_max_action < -1:
            self.next_max_action = np.array(-1).reshape(1)

    def expand(self):
        action_dict = {}
        self.get_ATP_limit()
        self.get_next_max_speed()
        self.get_max_acc_index()
        if self.next_max_action >= 0:
            for action in range(5 + int(abs(self.next_max_action) // 0.2)):
                if action not in self._children:
                    temp_action = self.action
                    self.action = self.next_max_action - 0.2 * action
                    self.reshape_action()
                    self.get_acc()
                    self.get_next_state()  # 需要通过状态转移知道expand出多少个节点
                    self.get_power()
                    self.get_simulate_reward()  # 这里需要完整的写一下说明文档
                    self._children[action] = MctsStateNode(self.next_state.copy(), self.step + 1, self.line, self.agent, self.episode, self.train_flag, self.train_model,
                                                           parent=self)  # parent=self是对的，因为要从此处开始进行演绎，删节点删到这个位置为止
                    self._children[action].current_reward = self.current_reward
                    self._children[action].last_node_action = self.action  # 这里应该用压缩过的工作，因为相当于开始MCTS的点采取了压缩后的动作才能expand
                    action_dict[action] = self.action
                    # 这里不应该对动作进行赋值了，因为要从_children[action]开始演绎，所以需要令其根据agent自己生成动作，而不是把父节点的动作给他
                    #  self._children[action].action = self.action
                    self.action = temp_action
        else:
            for action in range(5 - int(abs(self.next_max_action) // 0.2)):
                if action not in self._children:
                    temp_action = self.action
                    self.action = self.next_max_action - 0.2 * action
                    self.reshape_action()
                    self.get_acc()
                    self.get_next_state()
                    self.get_power()
                    self.get_simulate_reward()
                    self._children[action] = MctsStateNode(self.next_state.copy(), self.step + 1, self.line, self.agent, self.episode, self.train_flag, self.train_model, parent=self)
                    self._children[action].current_reward = self.current_reward
                    self._children[action].last_node_action = temp_action
                    action_dict[action] = self.action
                    # self._children[action].action = self.action
                    self.action = temp_action
        return action_dict

    def get_simulate_reward(self):
        self.speed_check()
        self.comfort_check()
        temp_time = self.line.delta_distance / (self.state[1] / 2 + self.next_state[1] / 2)
        if self.speed_punish:
            self.current_reward = -3 * self.t_power - 3 * self.re_power - 10.5 * abs(
                temp_time - 1.2 * (abs(self.line.scheduled_time - self.state[0]) / (self.max_step + 1 - self.step))) + self.p_indicator - 10 * self.comfort_punish  # 当前step的运行时间和剩余距离平均时间的差值
        else:
            self.current_reward = -3 * self.t_power - 3 * self.re_power - 10.5 * abs(
                1 * temp_time - 1.2 * (abs(self.line.scheduled_time - self.state[0]) / (self.max_step + 1 - self.step))) - 10 * self.comfort_punish

    def Mcts_State_Transition(self):
        self.get_ATP_limit()
        self.get_action()
        self.reshape_action()
        self.get_acc()
        if self.Mcts_Check() and self.step < self.max_step:
            temp_action = self.Mcts_Start()
            if temp_action is not None:
                self.action = temp_action
            else:
                self.get_safe_action()
                # self.action = np.array(-1.5).reshape(1)
            self.reshape_action()
            self.get_acc()
        self.get_next_state()
        self.get_power()

    def Mcts_State_Transition_eval(self):
        self.get_ATP_limit()
        self.get_action_eval()
        self.reshape_action()
        self.get_acc()
        if self.Mcts_Check() and self.step < self.max_step:
            temp_action = self.Mcts_Start()
            if temp_action is not None:
                self.action = temp_action
            else:
                self.get_safe_action()
                # self.action = np.array(-1.5).reshape(1)
            self.reshape_action()
            self.get_acc()
        self.get_next_state()
        self.get_power()

    def Mcts_State_Transition_eval_rob(self, noise_dict, noise_acc_dict, i, j):
        self.get_ATP_limit()
        self.get_action_eval()
        self.reshape_action()
        self.get_acc()
        if self.Mcts_Check() and self.step < self.max_step:
            temp_action = self.Mcts_Start()
            if temp_action is not None:
                self.action = temp_action
            else:
                self.get_safe_action()
                # self.action = np.array(-1.5).reshape(1)
            self.reshape_action()
            self.get_acc()
        random_ind = random.random()
        if random_ind <= 0.1 * i:
            o_t_action = self.action
            o_t_acc = self.acc.copy()
            noise_dict[self.step] = o_t_action
            noise_acc_dict[self.step] = o_t_acc
            random_noise = random.uniform(-0.1 * j, 0.1 * j)
            self.action = self.action + random_noise
            self.get_acc()
        self.get_next_state()
        self.get_power()
        return noise_dict, noise_acc_dict

    # 后面的是一般的方法

    def get_last_node(self, node_list):  # 获取上一个节点的状态、动作加速度
        if len(node_list) != 1:
            self.last_node_state = node_list[self.step - 1].state
            self.last_node_action = node_list[self.step - 1].action
            self.last_node_acc = node_list[self.step - 1].acc
        else:
            self.last_node_state = np.zeros(2)
            self.last_node_action = np.array(0).reshape(1)
            self.last_node_acc = 0

    # 下面是动作的产生过程
    def get_action_eval(self):  # 选择动作
        self.action = self.agent.choose_action(self.state)
        self.action = np.array(self.action).reshape(1)

    def get_action(self):  # 选择动作
        self.action, self.mean, self.log_std = self.agent.policy_net.get_action(self.state)
        self.action = np.array(self.action).reshape(1)

    def get_ATP_limit(self):
        key_list = []
        key = 0
        key_next = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]  # 当前限速点位置
                key_next = key_list[j + 1]  # 下一个限速点位置
        limit_speed = self.line.speed_limit[key]  # 当前位置限速
        limit_speed_next = self.line.speed_limit[key_next]  # 下一个限速点限速
        final_locate = key_next // self.line.delta_distance
        length = final_locate - self.step
        if limit_speed_next < limit_speed:
            self.ATP_limit = np.sqrt(
                limit_speed_next * limit_speed_next / 3.6 / 3.6 - 2 * 0.5 * self.train_model.max_bra_acc * length * self.line.delta_distance) * 3.6

    def reshape_action(self):  # 重整动作
        if self.step <= 0.001 * self.max_step:
            self.action = (self.action + 0.03) / 2
        elif self.max_step - self.step <= 0.0015 * self.max_step:
            self.action = (self.action - 1) / 1
        else:
            self.action = self.action
        low_bound = -1
        upper_bound = 1
        # 重整当前动作
        # self.action = low_bound + (self.action + 1.0) * 0.5 * (upper_bound - low_bound)
        self.action = np.clip(self.action, low_bound, upper_bound)
        # 重整上一个节点的动作，这句话好像可以不要
        # self.last_node_action = low_bound + (self.last_node_action + 1.0) * 0.5 * (upper_bound - low_bound)
        # self.last_node_action = np.clip(self.last_node_action, low_bound, upper_bound)

    # 下面是合加速度的计算过程
    def get_gradient_acc(self):  # 获取当前位置坡度加速度
        key_list = []
        key = 0
        for i in self.line.gradient.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        gradient = self.line.gradient[key]
        self.g_acc = -9.8 * gradient / 1000  # g_ acc = 9.8 * g /1000
        self.slope = gradient / 10
        del key_list

    def get_curve_acc(self):  # 获取当前位置曲率加速度
        key_list = []
        key = 0
        for i in self.line.curve.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        curve = self.line.curve[key]
        if curve != 0:
            self.c_acc = - 3 * 9.8 / (5 * curve)  # c_acc = 3g/5R
        else:
            self.c_acc = 0
        del key_list

    def get_current_tra_acc(self):  # 计算当前牵引加速度
        # self.train_model.get_max_traction_force(self.state[1] * 3.6)  # 当前车辆的最大牵引力
        tra_force = self.train_model.max_traction_force * self.action  # 当前输出的牵引力
        self.tm_acc = 1 * tra_force / self.train_model.weight

    def get_current_b_acc(self):  # 计算当前制动加速度
        # self.train_model.get_max_brake_force(self.state[1] * 3.6)
        bra_force = self.train_model.max_brake_force * abs(self.action)  # 单位是kN
        self.bm_acc = - 1 * bra_force / self.train_model.weight

    def get_m_acc(self):  # 判断当前是制动还是牵引
        self.train_model.get_max_traction_force(self.state[1] * 3.6)  # 当前车辆的最大牵引力
        self.train_model.get_max_brake_force(self.state[1] * 3.6)
        if self.action < 0:
            self.tm_acc = 0
            self.get_current_b_acc()
        else:
            self.get_current_tra_acc()
            self.bm_acc = 0

    def get_acc(self):
        self.get_m_acc()
        self.get_gradient_acc()
        self.get_curve_acc()
        self.acc = self.tm_acc + self.bm_acc + self.g_acc + self.c_acc
        if self.acc > 1.2:
            self.acc = np.array(1.2).reshape(1)
        if self.acc < -1.2:
            self.acc = np.array(-1.2).reshape(1)

    # 下面是能耗的计算过程
    def get_ave_v(self):  # 获取平均速度
        self.ave_v = 0.5 * (self.state[1] + self.next_state[1])

    def get_t_power(self):
        delta_t = self.next_state[0] - self.state[0]
        self.t_power = self.train_model.get_traction_power(self.ave_v * 3.6, delta_t, self.action)

    def get_re_power(self):
        delta_t = self.next_state[0] - self.state[0]
        self.re_power = self.train_model.get_re_power(self.ave_v * 3.6, delta_t, self.action)

    def get_power(self):
        self.get_ave_v()
        if self.action < 0:
            self.t_power = np.array(0.0).reshape(1)
            self.get_re_power()
        else:
            self.re_power = np.array(0.0).reshape(1)
            self.get_t_power()

    # 下面是超速检查过程
    def speed_check(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        limit_speed = self.line.speed_limit[key]
        if self.state[1] * 3.6 >= min(limit_speed - 2, self.ATP_limit):  # 超速
            self.speed_punish = 1
            self.current_limit_speed = min((limit_speed - 2) / 3.6, self.ATP_limit / 3.6)
        else:
            self.speed_punish = 0
            self.current_limit_speed = min((limit_speed - 2) / 3.6, self.ATP_limit / 3.6)

    # 下面是舒适度检查过程
    def comfort_check(self):
        if abs(self.acc - self.last_node_acc) >= 0.3:
            self.comfort_punish = 1.5  # 不舒适
        else:
            self.comfort_punish = 0

    # 下面是状态转移函数
    def get_next_state(self):
        time = np.array(self.state[0]).reshape(1)
        temp_velocity = np.array(self.state[1]).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        temp_time = self.line.delta_distance / (velocity / 2 + temp_velocity / 2)
        time = time + temp_time
        self.next_state[0] = time  # 状态转移后的时间
        self.next_state[1] = velocity  # 状态转移后的速度
        # self.next_state[2] = self.acc  # 用什么样的加速度达到下状态（上一个动作产生的加速度）
        self.next_state[2] = (self.step + 1) * self.line.delta_distance / 1000  # 下状态所处的位置，对于当前的状态就是当前位置
        # self.next_state[2] = self.acc

    # 下面是完整的一般状态转移过程
    def state_transition(self):
        self.get_action()
        self.reshape_action()
        self.get_acc()
        self.get_next_state()
        self.get_power()

    # 加入shield后的安全状态转移过程
    def safe_state_transition(self):
        self.get_ATP_limit()
        self.get_action()
        self.reshape_action()
        self.get_acc()
        if self.Shield_Check():
            self.get_safe_action()
            self.reshape_action()
            self.get_acc()
        self.get_next_state()
        self.get_power()

    # 判断是否需要使用Shield
    def Shield_Check(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        next_limit_speed = self.line.speed_limit[key]
        temp_velocity = np.array(self.state[1]).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        if velocity * 3.6 > min(next_limit_speed - 2, self.ATP_limit):
            return 1
        else:
            return 0

    def speed_check2(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        limit_speed = self.line.speed_limit[key]
        self.c_limit_speed = self.line.speed_limit[key]
        self.a_limit_speed = min(self.c_limit_speed, self.ATP_limit)
        if self.state[1] * 3.6 >= limit_speed:  # 超速
            self.speed_punish = 1
        else:
            self.speed_punish = 0

    # 获取安全动作的函数
    def get_safe_action(self):
        self.shield_flag = 1
        xunhuan_count = 0
        chaosu_flag = 0
        initial_velocity = self.state[1].copy()
        while chaosu_flag != 1:
            xunhuan_count += 1
            if xunhuan_count > 100:
                temp_acc = self.acc - self.g_acc - self.c_acc
                if temp_acc >= 0:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_traction_force
                else:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_brake_force
                if self.action > 1:
                    self.action = np.array(1).reshape(1)
                if self.action < -1:
                    self.action = np.array(-1).reshape(1)
                break
            # temp_velocity = self.state[1]
            temp_velocity = initial_velocity
            temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
            if temp_square_velocity <= 1:
                temp_square_velocity = np.array(1).reshape(1)
            velocity = np.sqrt(temp_square_velocity)
            if velocity <= 0:
                velocity = np.array(1).reshape(1)
            self.state[1] = velocity
            self.speed_check()
            if self.speed_punish:
                # self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                if xunhuan_count == 1:
                    if self.acc > 0:
                        # self.acc = self.acc - 0.2
                        self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                    else:
                        self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                else:
                    self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
            else:
                chaosu_flag = 1
                temp_acc = self.acc - self.g_acc - self.c_acc
                if temp_acc >= 0:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_traction_force
                else:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_brake_force
                if self.action > 1:
                    self.action = np.array(1).reshape(1)
                if self.action < -1:
                    self.action = np.array(-1).reshape(1)
        self.state[1] = initial_velocity

    # 下面是奖励的计算
    def get_reward(self, unsafe_counts, total_power, ep_protect_counts):
        self.speed_check2()
        self.comfort_check()
        if self.step == self.max_step:
            self.done = 1
            if abs(self.next_state[0] - self.line.scheduled_time) <= 10 and abs(self.next_state[1]) <= 3:
                e_reward = 100  # 准时且停下施加额外奖励
            else:
                e_reward = 0
            if self.line.scheduled_time - self.next_state[0] < -10:
                t_punish = -self.next_state[0] + self.line.scheduled_time  # 晚点超过10s的惩罚
            elif self.line.scheduled_time - self.next_state[0] > 0:
                t_punish = -self.line.scheduled_time + self.next_state[0]  # 早到的惩罚
            else:
                t_punish = -self.next_state[0] + self.line.scheduled_time  # 晚点10s之内的惩罚
            if self.speed_punish:
                unsafe_counts += 1
                # self.current_reward = -0.001 * total_power - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                self.current_reward = -3 * (total_power - self.line.ac_power) - 25 * self.next_state[
                    1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                # self.current_reward = -1 * (total_power - self.line.ac_power) - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish + self.p_indicator
            else:
                unsafe_counts += 0
                # self.current_reward = -0.001 * total_power - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                self.current_reward = -3 * (total_power - self.line.ac_power) - 25 * self.next_state[
                    1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                # self.current_reward = -1 * (total_power - self.line.ac_power) - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
        else:
            self.done = 0  # 能耗前的系数影响平化程度，时间项的系数影响整体的曲线形状
            temp_time = self.line.delta_distance / (self.state[1] / 2 + self.next_state[1] / 2)
            if self.speed_punish:
                unsafe_counts += 1
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - 3.4 * abs(1.3 * temp_time - (self.line.scheduled_time / (self.max_step + 1))) + self.p_indicator - 10 * self.comfort_punish
                self.current_reward = -3 * self.t_power - 3 * self.re_power - 10.5 * abs(
                    1 * temp_time - 1.2 * (abs(self.line.scheduled_time - self.state[0]) / (
                            self.max_step + 1 - self.step))) + self.p_indicator - 10 * self.comfort_punish  # 当前step的运行时间和剩余距离平均时间的差值
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - abs(1 * (
                #         2 * self.line.delta_distance * (self.max_step + 1 - self.step) / (abs(self.line.scheduled_time - self.state[0])) - self.state[
                #     1])) + self.p_indicator - 10 * self.comfort_punish  # 当前step速度和剩余平均速度的差值
            else:
                unsafe_counts += 0
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - 3.4 * abs(1.3 * temp_time - (self.line.scheduled_time / (self.max_step + 1))) - 10 * self.comfort_punish
                self.current_reward = -3 * self.t_power - 3 * self.re_power - 10.5 * abs(
                    1 * temp_time - 1.2 * (abs(self.line.scheduled_time - self.state[0]) / (
                            self.max_step + 1 - self.step))) - 10 * self.comfort_punish
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - abs(1 * (
                #         2 * self.line.delta_distance * (self.max_step + 1 - self.step) / (abs(self.line.scheduled_time - self.state[0])) - self.state[
                #     1])) - 10 * self.comfort_punish
        if self.shield_flag:
            # self.current_reward += -50
            ep_protect_counts += 1
        return self.done, unsafe_counts, ep_protect_counts
