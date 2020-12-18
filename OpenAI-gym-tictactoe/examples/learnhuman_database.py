#!/usr/bin/env python
import os
import sys
import random
import time
import logging
import json
import pymysql

from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import pandas as pd
import click
from tqdm import tqdm as _tqdm
tqdm = _tqdm

from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD
# from human_agent import HumanAgent
from base_agent import BaseAgent


DEFAULT_VALUE = 0
EPISODE_CNT = 17000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'best_td_agent.dat'
INITIAL_MODEL_FILE = 'initial_td_agent.dat'
INITIAL_MODEL_FILE2 = 'initial_td_agent2.dat'
LEARNED_MODEL = 'learn_model.dat'
EPSILON = 0.08
ALPHA = 0.4
CWD = os.path.dirname(os.path.abspath(__file__))


st_values = {}
st_visits = defaultdict(lambda: 0)

def remove_X(nstate):
    """Remove 'X' for insert to SQL database
       and return list of nstate witout 'X'
    """
    nstate = [item for i in nstate for item in i]
    while 'X' in nstate: nstate.remove('X') #remove 'X' from list and we can insert from
    return nstate

def reset_state_values():
    global st_values, st_visits
    st_values = {}
    st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):
    st_visits[state] += 1
    st_values[state] = value


def best_val_indices(values, fn):
    best = fn(values)
    if(fn == max and values.count(-1) == len(values)):
        print(" >>>>>> There is no possible way to win. Please change the starting position <<<<<< ")
    return [i for i, v in enumerate(values) if v == best]

class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark
        self.alpha = 0.4

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break
        """Print action to move Robot Arm"""
        # print(action) 
        return action
    
    def ask_value(self, state):
        """Returns value of given state.

        If state is not exists, set it as default value.

        Args:
            state (tuple): State.

        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            logging.debug("ask_value - new state {}".format(state))
            gstatus = check_game_status(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def backup(self, state, nstate, reward):
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        logging.debug("backup state {} nstate {} reward {}".
                      format(state, nstate, reward))

        val = self.ask_value(state)
        nval = self.ask_value(nstate)
        diff = nval - val
        val2 = val + self.alpha * diff

        logging.debug("  value from {:0.2f} to {:0.2f}".format(val, val2))
        set_state_value(state, val2)

class TDAgent(object):
    def __init__(self, mark, epsilon, alpha):
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_rate = 1.0
        # create db variable for sending to database.
        self.db_nstate = []
        self.db_nvalue = []
        self.db_choose = 0
        self.db_pick = 0

    def set_db(self, nstate, value, choose, pick):
        # nstate = [item for i in nstate for item in i]
        # while 'X' in nstate: nstate.remove('X') #remove 'X' from list and we can insert from
        self.db_nstate = remove_X(nstate)
        self.db_nvalue = value
        self.db_choose = choose
        self.db_pick = pick

        print(self.db_nstate)
        print(self.db_nvalue)
        print(self.db_choose)
        print(self.db_pick)
    
    def get_db(self):
        return self.db_nstate, self.db_nvalue, self.db_choose, self.db_pick

    def act(self, state, ava_actions):
        return self.egreedy_policy(state, ava_actions)

    def egreedy_policy(self, state, ava_actions):
        """Returns action by Epsilon greedy policy.

        Return random action with epsilon probability or best action.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        logging.debug("egreedy_policy for '{}'".format(self.mark))
        e = random.random()
        # print(e)
        if e < self.epsilon * self.episode_rate:
            # print("take random action")
            logging.debug("Explore with eps {}".format(self.epsilon))
            action = self.random_action(ava_actions)
        else:
            # print("take greedy action")
            logging.debug("Exploit with eps {}".format(self.epsilon))
            action = self.greedy_action(state, ava_actions)
        """Print action to move Robot Arm"""
        # print(action)
        return action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):
        """Return best action by current state value.

        Evaluate each action, select best one. Tie-breaking is random.

        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions

        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0
        coun_nstate = 0
        ava_values = []
        # temp for insert to db variable
        temp_db_nstate = []
        temp_db_nvalue = []
        temp_db_choose = 0 

        for action in ava_actions:
            nstate = after_action_state(state, action)
            nval = self.ask_value(nstate)
            # show next state and reward 
            print("Choice:"+str(coun_nstate)+". %s || Reward is %s" % (nstate, nval))
            temp_db_nstate.append(nstate)
            temp_db_nvalue.append(nval)
            # print("temp_db_nstate: %s" %(temp_db_nstate))
            # print("temp_db_nvalue: %s" %(temp_db_nvalue))

            coun_nstate += 1
            ava_values.append(nval)
            vcnt = st_visits[nstate]
            logging.debug("  nstate {} val {:0.2f} visits {}".
                          format(nstate, nval, vcnt))
        
        # select most right action for 'O' or 'X'
        if self.mark == 'O':
            indices = best_val_indices(ava_values, max)
            print("---> Machine Choose Maximum Reward in choice(s) %s" % (indices))

        else:
            indices = best_val_indices(ava_values, min)
            print("---> Machine Choose Minimum Reward in choice(s) %s" % (indices))
        
        # tie breaking by random choice
        aidx = random.choice(indices)
        logging.debug("greedy_action mark {} ava_values {} indices {} aidx {}".
                      format(self.mark, ava_values, indices, aidx))
        print("------> Machine Choose choice %s." % (aidx))
        action = ava_actions[aidx]
        print("---------> Machine pick at %s." % str(action+1))

        self.set_db(temp_db_nstate , temp_db_nvalue, aidx, action)
        return action

    def ask_value(self, state):
        """Returns value of given state.

        If state is not exists, set it as default value.

        Args:
            state (tuple): State.

        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            logging.debug("ask_value - new state {}".format(state))
            gstatus = check_game_status(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def backup(self, state, nstate, reward):
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        logging.debug("backup state {} nstate {} reward {}".
                      format(state, nstate, reward))

        val = self.ask_value(state)
        # print(val)
        nval = self.ask_value(nstate)
        # print(nval)
        diff = nval - val
        # print(diff)
        val2 = val + self.alpha * diff
        # print(val2)
        

        logging.debug("  value from {:0.2f} to {:0.2f}".format(val, val2))
        set_state_value(state, val2)


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    global tqdm

    set_log_level_by(verbose)
    if verbose > 0:
        tqdm = lambda x: x  # NOQA


def save_model(save_file, max_episode, epsilon, alpha):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, epsilon=epsilon,
                    alpha=alpha)
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in st_values.items():
            vcnt = st_visits[state]
            f.write('{}\t{:0.3f}\t{}\n'.format(state, value, vcnt))


def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            vcnt = eval(elms[2])
            st_values[state] = val
            st_visits[state] = vcnt
    return info

# Emma2541
# this fuction is for show case how machine learning (reinforcement learning) works.
# Show step of learning By fixed start position if Human pick the same position from last round.
@cli.command(help="Learn from human for showcase.")
@click.option('-e', '--epsilon', "epsilon", default=0, #prevent random exploring
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--save-file', default=INITIAL_MODEL_FILE2, show_default=True,
              help="Save model data as file name.")
@click.option('-f', '--load-file', default=INITIAL_MODEL_FILE, show_default=True,
              help="Load file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def learnhuman1(epsilon, alpha, save_file, load_file, show_number):
    _learnhuman1(epsilon, alpha, save_file, load_file, HumanAgent('X'), show_number)

def _learnhuman1(epsilon, alpha, save_file, load_file, vs_agent, show_number):
    connection = pymysql.connect(host="localhost",user="root",passwd="",database="tictactoe" )
    cursor = connection.cursor()

    load_model(load_file)
    env = TicTacToeEnv(show_number=show_number)    
    start_mark = 'X'
    agents = [vs_agent, TDAgent('O', epsilon, alpha)]
    max_episode = 0
    agent_temp = 6 #Set Start position at 6 to td_agent
    human_temp = 0
    while True:
        # start agent rotation
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        turns = 0
        human_diff = False

        # variable for save to database
        db_nstate = []
        db_nvalue = []
        db_choose = 0
        db_pick = 0
        db_current_state = ''
        db_action = 0
        db_note = ''

        # show start board for human agent
        if mark == 'X':
            env.render(mode='human')

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)
            print("======================================Switch Turn======================================")
            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            # print(ava_actions)
            if human:
                action = agent.act(ava_actions)
                turns += 1
                if turns == 1 and human_temp != action:
                    human_temp = action
                    human_diff = True
                    # print("Human action is %s"%(action))
                    # print("Turns == %s" % (turns))
                    # print("human_temp == %s" %(human_temp))
                    # print("human_diff == %s" %(human_diff))
                if action is None:
                    sys.exit()
                nstate, reward, done, info = env.step(action)
            else:
                action = agent.act(state, ava_actions)
                db_nstate, db_nvalue, db_choose, db_pick = agent.get_db()
                if turns == 1 and human_diff == False:
                    action = agent_temp
                    db_note = 'repeat'
                    print("------------> Human start in the same position")
                    print("------------> [Fix]Agent action is %s"%(action+1))
                elif turns == 1 and human_diff == True:
                    agent_temp = action
                    db_note = 'different'
                    print("------------> Human start in the diiferent position")
                    print("------------> [New]Agent action is %s"%(action+1))
                else:
                    print("------------> Agent action is %s"%(action+1))
                nstate, reward, done, info = env.step(action)
                db_table_tictactoe_state = "INSERT INTO Tictactoe_state(EPISODE, NSTATE, NVALUE, CHOOSE, PICK, STATE_NOW, ACTION, NOTE) \
                    VALUES(%d, '%s', '%s', %d, %d, '%s', %d, '%s');" % (max_episode, db_nstate, db_nvalue, db_choose, db_pick, remove_X(nstate), action, db_note)
                cursor.execute(db_table_tictactoe_state)
                connection.commit()
            ### 
            agent.backup(state, nstate, reward)
            env.render(mode='human')
            if done:
                db_win_state = ''
                if reward == 0:
                   db_win_state = "draw"
                else:
                    db_win_state = mark
                print("Return reward : "+ str(reward))
                db_table_check_win = "INSERT INTO Check_win(REWARD, WIN_STATE) VALUES(%d, '%s');" % (reward, db_win_state)
                cursor.execute(db_table_check_win)
                connection.commit()
                env.show_result(True, mark, reward)
                time.sleep(1)
                # if reward == 1:
                    # _conlearn(700, epsilon, alpha, save_file, load_file)
                # set terminal state value
                set_state_value(state, reward)
                break
            else:
                _, mark = state = nstate
                connection.commit()

            # insert2 = "INSERT INTO Tictactoe_state(EPISODE, NSTATE, REWARD, ALL_CHOICE, CHOOSE, PICK, STATE_NOW, \
            #     ACTION, NOTE) VALUES(1, 'asdasd', 1, 'asdasd', 1, 1, 'asdasd', 1, 'asdasasd');"
            # cursor.execute(insert2)
            # connection.commit()
        # rotation start
        # start_mark = next_mark(start_mark)
        
        max_episode += 1
        # print(max_episode)
        save_model(save_file, max_episode, epsilon, alpha)

    connection.close()

if __name__ == '__main__':
    cli()
