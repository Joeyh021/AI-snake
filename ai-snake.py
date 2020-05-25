from os import environ  # set environment variables
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # shut pygame up
environ["CUDA_VISIBLE_DEVICES"] = "-1"  # dont use gpu
environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # shut tensorflow up (mostly)

import pygame
import numpy as np
from tensorflow import keras
from collections import deque
from datetime import datetime
from time import sleep
import json
import pickle as pkl

SCREEN_SIZE = 600
ROWS = 30
FONT = "comicsansms"
FONT_SIZE = 18
APPLE_REWARD = 100
DEATH_PENTALTY = -10
HUMAN_SPEED = 10
AI_SPEED = 100
DISCOUNT_FACTOR = 0.995
L_FILE = "leaderboard.json"
M_FILE = "AI-final.h5"
M_RIGHT = ((0, -1),
           (1, 0))
M_LEFT = ((0, 1),
          (-1, 0))  # constants for rotating vectors, appears wrong way because y axis is flipped but trust me its fine


class Grid:
    def __init__(self, display, rows):
        self.rows = rows
        self.matrix = np.zeros((self.rows, self.rows))

        self.display = display
        self.screen_size = self.display.get_width()
        self.draw()

    def draw(self):
        for i in range(self.rows):
            start = (self.screen_size / self.rows * i, 0)
            end = (self.screen_size / self.rows * i, self.screen_size)
            pygame.draw.line(self.display, (0, 0, 0), start, end)
        for i in range(self.rows):
            start = (0, self.screen_size / self.rows * i)
            end = (self.screen_size, self.screen_size / self.rows * i)
            pygame.draw.line(self.display, (0, 0, 0), start, end)

    def update(self):
        for i in range(self.rows):
            for j in range(self.rows):
                colour = self.matrix[i][j]
                if colour == 0:  # clear
                    colour = (255, 255, 255)
                elif colour == 1:  # body
                    colour = (0, 255, 0)
                elif colour == 2:  # head
                    colour = (0, 128, 0)
                elif colour == 4:  # apple
                    colour = (255, 0, 0)
                pygame.draw.rect(self.display, colour, (self.screen_size / self.rows * i + 1, self.screen_size / self.rows * j + 1, self.screen_size / self.rows - 1, self.screen_size / self.rows - 1))

    def clear(self):
        self.matrix = np.zeros((self.rows, self.rows))

    def setsq(self, xy, val):
        self.matrix[xy] = val

    def getval(self, xy):
        if (xy[0] < 0) or (xy[1] < 0) or (xy[0] > self.rows - 1) or (xy[1] > self.rows - 1):
            return -1
        else:
            return int(self.matrix[xy])


class Snake:
    def __init__(self, grid):
        self.grid = grid  # grid object
        self.pos = (int(grid.rows / 2), int(grid.rows / 2))  # head coordinate
        self.tail_len = 3
        self.body = []  # coords of body squares

    def draw(self):
        self.grid.setsq(self.pos, 2)  # head
        for i in self.body:
            self.grid.setsq(i, 1)  # body

    def move(self, direction):
        self.body.append(tuple(self.pos))  # add current position to body
        self.body = self.body[-(self.tail_len):]  # trim body to correct length
        self.pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])  # move


class Apple:
    def __init__(self, grid):
        self.grid = grid
        self.pos = (0, 0)
        self.repos()  # randomly position on creation

    def repos(self):
        # randomly generate coordinates
        self.pos = (np.random.randint(0, self.grid.rows), np.random.randint(0, self.grid.rows))
        if self.grid.getval(self.pos) != 0:  # if randomly selected coords have snake in it
            self.repos()  # recursion ooooh

    def draw(self):
        self.grid.setsq(self.pos, 4)


class Game:
    def __init__(self, rows, screen_size):
        # initialise pygame, window and fonts and start new game
        pygame.init()
        self.display = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("snek")
        self.display.fill((255, 255, 255))

        pygame.font.init()
        self.font = pygame.font.SysFont(FONT, FONT_SIZE)
        self.grid = Grid(self.display, rows)
        self.new_game()

    # function to return state of environment, for AI to use
    def get_state(self):

        item_left = self.grid.getval((self.snake.pos[0] + self.direction[1], self.snake.pos[1] - self.direction[0]))
        item_right = self.grid.getval((self.snake.pos[0] - self.direction[1], self.snake.pos[1] + self.direction[0]))
        item_front = self.grid.getval((self.snake.pos[0] + self.direction[0], self.snake.pos[1] + self.direction[1]))

        if item_left == 0:
            obstacle_left = 0
        elif item_left == 4:
            obstacle_left = 1
        else:
            obstacle_left = -1

        if item_right == 0:
            obstacle_right = 0
        elif item_right == 4:
            obstacle_right = 1
        else:
            obstacle_right = -1

        if item_front == 0:
            obstacle_front = 0
        elif item_front == 4:
            obstacle_front = 1
        else:
            obstacle_front = -1

        # position of apple relative to snake x and y
        pos_x = (self.snake.pos[0] - self.apple.pos[0])

        pos_y = (self.apple.pos[1] - self.snake.pos[1])

        # state is returned as tuple
        return (obstacle_left, obstacle_right, obstacle_front, pos_x, pos_y, self.direction[0], -self.direction[1], self.snake.tail_len)

    def draw_score(self):  # draw the score onto the screen
        self.display.blit(self.font.render(str(self.score), True, (0, 0, 0)), (5, -2))

    def new_game(self):
        self.grid.clear()
        self.score = 0
        self.done = False
        self.direction = (0, 1)
        self.prev_direction = self.direction
        self.snake = Snake(self.grid)
        self.snake.draw()
        self.apple = Apple(self.grid)
        self.apple.draw()
        self.grid.update()
        pygame.display.update()

    # function to progress the game by one step, based upon action passed to it
    def step(self, action):
        if action == 0:  # carry on
            self.direction = self.prev_direction
        elif action == 1:  # turn right
            self.direction = tuple(np.dot(M_RIGHT, self.direction))  # (-self.prev_direction[1], self.prev_direction[0])
        elif action == 2:  # turn left
            self.direction = tuple(np.dot(M_LEFT, self.direction))  # (self.prev_direction[1], -self.prev_direction[0])

        self.snake.move(self.direction)
        self.prev_direction = self.direction  # change snake direction and move

        # check if snake died based upon the movement taken
        if self.grid.getval(self.snake.pos) == -1:
            self.done = True
        else:
            for n in self.snake.body:
                if n == self.snake.pos:
                    self.done = True

        # if dead, reset game
        apple_got = False
        if self.done:
            return (apple_got, True, self.score)

        # if snake at the apple, make a new apple

        if self.snake.pos == self.apple.pos:
            self.apple.repos()
            self.score += 1
            self.snake.tail_len += 1
            apple_got = True

        self.grid.clear()
        self.snake.draw()
        self.apple.draw()
        self.grid.update()
        self.draw_score()
        pygame.display.update()  # draw and update everything

        return (apple_got, self.done, self.score)


class play_game(Game):
    def __init__(self, rows, screen_size, speed):
        super().__init__(rows, screen_size)
        self.speed = speed

    def new_game(self):
        super().new_game()
        self.clock = pygame.time.Clock()

    def step(self, action):
        a = super().step(action)
        self.clock.tick(self.speed)
        return a


class Leaderboard:
    def __init__(self, filename):
        self.filename = filename
        try:
            with open(self.filename, "r") as f:
                self.scores = json.load(f)
        except FileNotFoundError:
            with open(self.filename, "w") as f:
                json.dump({"all": []}, f)
            with open(self.filename, "r") as f:
                self.scores = json.load(f)

    def test(self, name, score):  # returns True if score worth adding
        if name not in self.scores:  # create them if they dont exist
            return True
        elif len(self.scores["all"]) < 20:  # if less than 20 scores exist add it
            return True
        elif len(self.scores[name]) < 10:  # if less than 10 scores exist for them add it
            return True
        elif self.scores["all"][19][1] < score:  # if the score is higher than the bottom of the leaderboard add it
            return True
        elif self.scores[name][9][0] < score:  # if the score is higher than the bottom of their leaderboard add it
            return True
        else:
            return False  # else just dont bother

    def add(self, name, score):
        name = name.lower()
        if name == "all":
            raise ValueError("name cant be 'all'")
        # add to personal scores list
        if self.test(name, score):
            time = datetime.now().strftime("%d/%m/%y, %H:%M ")
            if name in self.scores:
                self.scores[name].append((score, time))  # add
                self.scores[name].sort(key=lambda x: x[0], reverse=True)  # sort
                self.scores[name] = self.scores[name][:10]  # chop
            else:
                self.scores[name] = [[score, time]]

            # add to high scores
            self.scores["all"].append((name, score, time))  # add
            self.scores["all"].sort(key=lambda x: x[1], reverse=True)  # sort
            self.scores["all"] = self.scores["all"][:20]  # chop

    def display(self, name="all"):
        name = name.lower()
        if name not in self.scores:
            raise ValueError("That player does not exist")
        if len(self.scores[name]) == 0:
            print("No Entries!")
            return
        if name == "all":
            print(" ---- Top Scores All Time ----")
            for n, s, d in self.scores["all"]:
                print("{:<5} - {:<2d} - {}".format(n, s, d))
        else:
            print(" ---- Top Scores For " + name.capitalize() + " ----")
            for s, d in self.scores[name]:
                print("{:<2d} - {}".format(s, d))

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.scores, f)


class DQPlayer:
    def __init__(self, filename):
        self.net = keras.models.load_model(filename)
        self.l = Leaderboard(L_FILE)
        self.loop_threshold = 300

    def act(self, state):
        Q_vals = self.net.predict(np.array([state]))[0]  # pass state to net
        action = np.argmax(Q_vals)  # pick largest Q-val
        return action, list(Q_vals)

    def play(self, n_episodes):
        env = play_game(ROWS, SCREEN_SIZE, AI_SPEED)
        tracker = []
        for e in range(n_episodes):
            dead = False
            state = env.get_state()
            steps = 0
            prev_score = 0
            steps_since_score = 0
            second_Qs = []
            target_Q = None
            while not dead:  # main game loop
                pygame.event.pump()
                action, Q_vals = self.act(state)

                if target_Q in Q_vals:
                    action = Q_vals.index(target_Q)
                    target_Q = None  # reset loop detection
                    second_Qs = []
                _, dead, score = env.step(action)

                if score == prev_score:
                    steps_since_score += 1
                else:
                    steps_since_score = 0
                prev_score = score

                state = env.get_state()
                steps += 1

                if steps_since_score > self.loop_threshold:
                    second_Qs.append(sorted(Q_vals)[1])  # get second highest Q-value
                    if steps_since_score > self.loop_threshold + 100:
                        target_Q = np.max(second_Qs)
                if steps_since_score > self.loop_threshold * 5:
                    dead = True

            tracker.append((score, steps))
            print("episode {:<3d} survived {:<4d} steps and scored {:<2d}".format(e, steps, score))
            env.new_game()
            self.l.add("AI", score)

        print(str(n_episodes) + " episodes complete")

        tracker = np.array(tracker)
        score, steps = np.hsplit(tracker, 2)
        print("max score of {:2d}".format(np.max(score)))
        print("mean score of {:2.1f}".format(np.mean(score)))
        print("median score of {:2.1f}".format(np.median(score)))
        self.l.save()


class HumanPlayer:
    def __init__(self):
        self.game = play_game(ROWS, SCREEN_SIZE, HUMAN_SPEED)
        self.score = 0

    def game_over(self):
        self.game.display.blit(self.game.font.render("G A M E   O V E R", True, (0, 0, 0)), (230, 220))
        pygame.display.update()
        sleep(2)
        self.game.display.blit(self.game.font.render("Score - " + str(self.score), True, (0, 0, 0)), (255, 250))
        pygame.display.update()
        sleep(2)
        pygame.quit()
        name = input("enter your name >>>")
        l = Leaderboard(L_FILE)
        l.add(name, self.score)
        l.save()

    def play(self):
        dead = False
        keypress = None
        while not dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # event handling
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        keypress = (-1, 0)
                    if event.key == pygame.K_d:
                        keypress = (1, 0)
                    if event.key == pygame.K_w:
                        keypress = (0, -1)
                    if event.key == pygame.K_s:
                        keypress = (0, 1)

            # convert keypress to direction
            left = tuple(np.dot(M_LEFT, self.game.direction))
            right = tuple(np.dot(M_RIGHT, self.game.direction))
            if keypress == right:
                action = 1
            elif keypress == left:
                action = 2
            else:
                action = 0

            _, dead, self.score = self.game.step(action)
            if dead:
                self.game_over()


class DQTrainer:
    def __init__(self, model=None):
        self.state_size = 8
        self.n_actions = 3

        self.gamma = DISCOUNT_FACTOR  # discount factor
        # e-greedy
        self.e_decay = -0.00013

        self.batch_size = 64
        self.min_mem_length = 1000
        # replay memory deque
        self.replay_mem = deque(maxlen=10000)

        if model:
            self.epsilon = 0
            self.active_net = keras.models.load_model(model)
        else:
            self.epsilon = 1
            self.active_net = self.build_net()

        self.target_net = self.build_net()

    def build_net(self):
        # define the network architecture for Keras
        net = keras.models.Sequential()
        net.add(keras.layers.Dense(12, input_dim=self.state_size, activation="relu"))
        net.add(keras.layers.Dense(self.n_actions, activation="linear"))
        net.compile(loss="mse", optimizer="Adam")
        return net

    def update_target(self):
        # update the target network to be equal to the active network
        self.target_net.set_weights(self.active_net.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # add an action taken to the memory, and decay the epsilon to increase exploitation with each action taken
        self.replay_mem.append((state, action, reward, next_state, done))
        self.epsilon *= np.exp(self.e_decay)

    def act(self, state):
        # choose an action either randomly or by network
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            Q_vals = self.active_net.predict(np.array([state]))[0]
            action = np.argmax(Q_vals)
        return action

    def learn(self):
        if len(self.replay_mem) > self.min_mem_length:  # only run if enough data gathered

            # grab a random sample and unpack it into lists and arrays
            batch = np.array([self.replay_mem[x] for x in np.random.randint(len(self.replay_mem), size=self.batch_size)])

            states = np.zeros((self.batch_size, self.state_size))
            next_states = np.zeros((self.batch_size, self.state_size))
            actions, rewards, dones = [], [], []

            for n in range(self.batch_size):
                states[n] = batch[n, 0]
                actions.append(batch[n, 1])
                rewards.append(batch[n, 2])
                next_states[n] = batch[n, 3]
                dones.append(batch[n, 4])

            # predict Q values of actions from s from active net
            target = self.active_net.predict(states)
            # predict Q values of actions from s' from active net
            target_next_active = self.active_net.predict(next_states)
            # predict Q values of actions from s' from target net
            target_next_target = self.target_net.predict(next_states)

            for n in range(self.batch_size):
                # get maximum Q value at s' from target model
                if dones[n]:
                    target[n][actions[n]] = rewards[n]
                    # if action directly caused a failure give it its negative reward instead of predicted one, so network learns not to fail
                    # this can be done in this case as there will be no more future rewards, so bellman equation below doesnt apply
                    # cannot be done for actions that caused positive rewards as future after that has to be taken into account

                else:
                    a = np.argmax(target_next_active[n])  # select most favourable action from s' using active net
                    target[n][actions[n]] = rewards[n] + self.gamma * target_next_target[n][a]
                    # set Q value for action taken equal to max discounted future reward from target net
                    # target_next_target[n][a] is the Q-value for the action taken from s'
                    # this makes target[n][actions[n]] the Q-value for the action taken from s, equal to the known reward plus estimated future reward

            # train the batch, fit states to target Q values
            self.active_net.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def train(self, n_episodes, max_steps=1000):
        env = Game(ROWS, SCREEN_SIZE)
        data = []
        for e in range(n_episodes):
            dead = False
            score = 0
            state = env.get_state()
            steps = 0

            # play episode
            while True:
                pygame.event.pump()

                action = self.act(state)  # choose an action and take it
                apple, dead, _ = env.step(action)
                if dead:
                    reward = DEATH_PENTALTY
                elif apple:
                    reward = APPLE_REWARD
                    score += 1
                else:
                    reward = 0
                next_state = env.get_state()

                self.remember(state, action, reward, next_state, dead)

                state = next_state
                steps += 1  # move to next step

                self.learn()  # train model at each step

                if dead or (steps >= max_steps):
                    self.update_target()  # copy the active network to the target network
                    env.new_game()  # start new game
                    print("episode {0:<4d} survived {1:<4d} steps scored {2:<2d} memory len {3:<4d} epsilon {4:<6.9f}"
                          .format(e, steps, score, len(self.replay_mem), self.epsilon))
                    data.append((e, steps, score, self.epsilon))
                    break

        # save trained model
        print("{} episodes complete, saving model...".format(n_episodes))
        timestamp = datetime.now().strftime("%d-%m-%H-%M")
        self.target_net.save("model-" + timestamp + ".h5")
        with open("data-{}.pkl".format(timestamp), "wb") as f:
            pkl.dump(data, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Snake")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", type=int, metavar="n_games", nargs="?", const="1000",
                       help="train a new model, specify number of games to train with, default 10000")
    group.add_argument("-r", "--run", type=int, metavar="n_games", nargs="?", const="1",
                       help="run some games with a trained model, specify number of games to run, default 3")
    group.add_argument("-p", "--play", action="store_true", help="play the game yourself")
    group.add_argument("-l", "--leaderboard", type=str, metavar="player name", nargs="?", const="all",
                       help="view the leaderboard,specify players scores to view, displays all by default")
    group.add_argument("-d", "--debug", action="store_true", help="run in interactive mode for debugging")
    args = parser.parse_args()
    if args.play:
        p = HumanPlayer()
        p.play()
    elif args.leaderboard:
        l = Leaderboard(L_FILE)
        try:
            l.display(name=args.leaderboard)
        except ValueError:
            print("that player does not exist!")
    elif args.run:
        r = DQPlayer(M_FILE)
        r.play(args.run)
    elif args.train:
        t = DQTrainer()
        t.train(args.train)
    elif args.debug:
        import code
        code.interact(local=locals())
