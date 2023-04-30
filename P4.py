import requests
import json
import numpy as np
import random
import time
import pickle
import datetime

# world: 40*40
# limit: step limit, or rewards all taken

class QLearner:
    def __init__(self, n, m, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.95):
        self.actions = ['N', 'S', 'E', 'W']
        self.q_table = np.zeros((n, m, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        # random exploration rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # drop rate of epsilon
        self.epsilon_decay = epsilon_decay

    # P(choose randomly) = self.epsilon, else choose based on q table
    def choose_action(self, state):
        # get the current date and time
        now = datetime.datetime.now()

        # format the date and time into a string
        date_string = now.strftime("%m-%d-%H-%M-%S")

        print(f'[{date_string}]', end = '')
        if random.uniform(0, 1) < self.epsilon:
            print('R ', end = '')
            return random.choice(self.actions)
        else:
            print('Q ', end = '')
            return self.actions[np.argmax(self.q_table[state])]

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state + (self.actions.index(action),)]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state + (self.actions.index(action),)] = new_q

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        data = {'q_table': self.q_table, 'epsilon': self.epsilon}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved file {filename}. ')

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
        print(f'Loaded file {filename}. Epsilon = {self.epsilon}')

def train(num_episodes, n, m, worldid, model_file=None):
    q_learner = QLearner(n, m)
    last_call_time = None

    if model_file:
        try:
            q_learner.load(model_file)
        except FileNotFoundError:
            print(f"Model file '{model_file}' not found. Starting training from scratch.")
    else:
        model_file=f'./q_data/q_table{worldid}.pkl'

    for episode in range(num_episodes):
        _,state = get_location()
        if not state:
            _, _, state, last_call_time = enter_world(worldid, last_call_time)

        total_reward = 0
        num_steps = 0
        #collected_rewards = 0

        while 1:
            action = q_learner.choose_action(state)
            try:
                print(f'E {episode} M {num_steps}: ', end = '')
                _, _, reward, scoreIncrement, next_state = make_move(worldid, action)
            except Exception as e:
                print(f"Terminal state reached in episode {episode}: {e}\nSteps: {num_steps}\nTotal reward: {total_reward}\nEpsilon: {q_learner.epsilon}")
                avg_reward = total_reward / num_steps
                print(f'Episode {episode + 1}: Average reward = {avg_reward}, Epsilon = {q_learner.epsilon}')
                q_learner.update_epsilon()
                q_learner.save(model_file)
                break


            q_learner.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            num_steps += 1
            if num_steps % 50 == 0:
                print(f'Epsilon = {q_learner.epsilon}')
                q_learner.save(model_file)
            time.sleep(0.1)

            if num_steps>5000:
                _,state = get_location()
                if state:
                    print(f"Terminal state reached in episode {episode}: \nSteps: {num_steps}\nTotal reward: {total_reward}\nEpsilon: {q_learner.epsilon}")
                    avg_reward = total_reward / num_steps
                    print(f'Episode {episode + 1}: Average reward = {avg_reward}, Epsilon = {q_learner.epsilon}')
                    q_learner.update_epsilon()
                    q_learner.save(model_file)
                    reset()
                    break



################# APIs below ################

def get_location():
    tail = f'?type=location&teamId={my_teamid}'
    try:
        response = requests.request("GET", url + tail, headers=headers, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        world = json_data["world"]
        state = json_data["state"]
        if world == '-1':
            print(f'Got my location: world {world}, state {state}. ')
            return world, state
        state = state.split(':')
        state = (int(state[0]), int(state[1]))
        print(f'Got my location: world {world}, state {state}. ')
        return world, state
    else:
        raise Exception(f'ERROR: failed to get my location. \nresponse: {response.text}')

def get_runs(count):
    tail = f'?type=runs&teamId={my_teamid}&count={count}'
    try:
        response = requests.request("GET", score_url + tail, headers=headers, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        runs = json_data["runs"]
        print(f'Got my last {count} runs: {runs}. ')
        return runs
    else:
        raise Exception(f'ERROR: failed to get my last {count} runs. \nresponse: {response.text}')

def enter_world(worldid, time_ = None):
    if time_ != None:
        time_passed = time.time() - time_
        if time_passed < 600:
            print(f'Less than 10 min from last Enter World operation, sleeping {600 - time_passed}s.')
            time.sleep(600 - time_passed)

    payload={'type': 'enter',
        'worldId': str(worldid),
        'teamId': my_teamid}

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    #print(response.text)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        worldid_ = json_data["worldId"]
        runid = json_data["runId"]
        state = json_data["state"]
        if worldid_ == '-1':
            print(f'Entered world {worldid_}, runid: {runid}, state: {state}. ')
            return worldid_, runid, state
        state = state.split(':')
        state = (int(state[0]), int(state[1]))
        print(f'Entered world {worldid_}, runid: {runid}, state: {state}. ')
        return worldid_, runid, state, time.time()
    else:
        raise Exception(f'ERROR: failed to enter world {worldid}. \nresponse: {response.text}')

def make_move(worldid, move, retry = 0):
    payload={'type': 'move',
        'move': move,
        'teamId': my_teamid,
        'worldId': str(worldid)}

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        if retry < 6:
            time.sleep(1)
            print(f'Retry: {retry}')
            return make_move(worldid, move, retry = retry+1)
        else:
            print('Retry too much times, exiting...')
            exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    #print(response.text)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        worldid_ = json_data["worldId"]
        runid = json_data["runId"]
        reward = json_data["reward"]
        scoreIncrement = json_data["scoreIncrement"]
        newState = json_data["newState"]
        newState = (int(newState['x']), int(newState['y']))
        print(f'Made move {move} in world {worldid_}, runid: {runid}, reward: {reward}, scoreIncrement: {scoreIncrement}, newState: {newState}. ')
        return worldid_, runid, reward, scoreIncrement, newState
    else:
        raise Exception(f'ERROR: failed to make move {move} in world {worldid}. \nresponse: {response.text}')

def get_score(count):
    tail = f'?type=score&teamId={my_teamid}'
    try:
        response = requests.request("GET", score_url + tail, headers=headers, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        score = json_data["score"]
        print(f'Got my score: {score}. ')
        return score
    else:
        raise Exception(f'ERROR: failed to get my score. \nresponse: {response.text}')

def reset():
    reset_url = f'https://www.notexponential.com/aip2pgaming/api/rl/reset.php?teamId={my_teamid}&otp=5712768807'
    try:
        response = requests.request("GET", reset_url, headers=headers, timeout=request_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out.")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        exit(1)

    json_data = json.loads(response.text)
    if json_data["code"] in ('OK', 'ok', 'Ok'):
        team_id = json_data["teamId"]
        print(f'Reset team {team_id} successfully. ')
    else:
        raise Exception(f'ERROR: failed to reset team {my_teamid}. \nresponse: {response.text}')

def main():
    # personal key and id
    try:
        with open('config.txt', 'r') as f:
            api_key = f.readline().strip()
            userid = f.readline().strip()
    except:
        print('Store your api_key and userid in config.txt in current path. \nFirst line: api_key, second line: userid. No variable name or quotation marks. ')
    print('api_key =',api_key)
    print('userid =',userid)
    #return

    global request_timeout_seconds
    request_timeout_seconds = 10

    global my_teamid
    my_teamid = '1351'

    global url
    url = "https://www.notexponential.com/aip2pgaming/api/rl/gw.php"

    global score_url
    score_url = 'https://www.notexponential.com/aip2pgaming/api/rl/score.php'

    # added 'User-Agent' item to avoid being blocked by Mod_Security
    global headers
    headers = {
        'x-api-key': api_key,
        'userid': userid,
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }

    # world,state = get_location()
    # print(type(world), type(state))
    # print(world,state)

    world = 10

    
    model_file = f'./q_data/q_table{world}.pkl'
    q_table = train(5, 40, 40, world, model_file)


if __name__ == '__main__':
    main()
