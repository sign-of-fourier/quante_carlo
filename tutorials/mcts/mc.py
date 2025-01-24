import numpy as np
import random 
import json
import requests

class GoGame:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.ko = None
        self.n_moves = 0
        self.n_captured = [0, 0, 0]
        self.parent = None
        self.left = None
        self.right = None

    def place_stone(self, x, y):
        # todo: remove move from list of available moves
        
        if self.is_valid_move(x, y):
            self.board[x, y] = self.current_player
            self.capture_stones(x, y)
            self.current_player = 3 - self.current_player
            self.n_moves += 1
            return True
        return False

    def is_valid_move(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x, y] != 0:
            return False
        if (x, y) == self.ko: # what's this do?
            return False
        return True

    def capture_stones(self, x, y):
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.board[nx, ny] == 3 - self.current_player:
                    group = self.find_group(nx, ny)
                    if self.count_liberties(group) == 0:
                        for gx, gy in group:
                            self.board[gx, gy] = 0
                            self.n_captured[3 - self.current_player] += 1
#                        print("{} {} captured".format(self.n_captured[1], self.n_captured[2]))
        # add move to list of available moves
        
    def find_group(self, x, y):
        color = self.board[x, y]
        group = set([(x, y)])
        frontier = [(x, y)]
        while frontier:
            cx, cy = frontier.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == color and (nx, ny) not in group:
                        group.add((nx, ny))
                        frontier.append((nx, ny))
        return group

    def count_liberties(self, group):
        liberties = set()
        for x, y in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == 0:
                        liberties.add((nx, ny))
        return len(liberties)

    def print_board(self):
        for row in self.board:
            print(' '.join(['.OX'[cell] for cell in row]))
        print()
        
    def get_random_move(self):

        if self.finished():
            print("No moves left")
            return -1, -1
        else:
            move_x, move_y = random.randint(0,8), random.randint(0,8)
            while not self.is_valid_move(move_x, move_y):
                move_x, move_y = random.randint(0,8), random.randint(0,8)
            return move_x, move_y

    def copy(self):
        c = GoGame(self.size)
        c.board = self.board.copy()
        c.current_player = self.current_player
        c.n_moves = self.n_moves
        c.n_captured = self.n_captured.copy()
        return c

    def finished(self):
        return (self.n_moves - sum(self.n_captured)) >= self.size**2
        



def minimax_wp(x, y, player):
    if player == 1:
        return x.mu > y.mu
    else:
        return x.mu < y.mu


class Node:
    def __init__(self, game, x, y):
        self.game = game
        self.scores = []
        self.game_embedding = game.board.flatten()
        self.children = []
        self.x = x
        self.y = y
        self.mu = -1
        self.possible_next_moves = []

    def print(self):
        if len(self.children) > 0:
            return [x.print() for x in self.children]
        else:
            return '*'
    
    def expand(self, n_children):
        
        for moves in range(n_children):
            x, y = self.game.get_random_move()
            self.children.append(Node(self.game.copy(), x, y))
            self.children[-1].game.place_stone(x, y)
            self.children[-1].parent = self
        
    def expand_single(self, x, y):
        
        self.children.append(Node(self.game.copy(), x, y))
        self.children[-1].game.place_stone(x, y)
        self.children[-1].parent = self
        
        return self.children
        
    def MC(self, N):

        scores = []
        for n in range(N):
            first_move = True
            mc_clone = self.game.copy()
            while (mc_clone.n_moves - np.sum(mc_clone.n_captured)) < mc_clone.size**2:
                x, y = mc_clone.get_random_move()
                mc_clone.place_stone(x, y)
                if first_move:
                    
                    # Possible Next Moves are not MC'd
                    
                    self.possible_next_moves.append(Node(mc_clone.copy(), x, y))
                    self.possible_next_moves[-1].parent = self
                    first_move = False
                    
                if (mc_clone.n_moves - np.sum(mc_clone.n_captured)) < mc_clone.size**2:    
                    x, y = mc_clone.get_random_move()
                    mc_clone.place_stone(x, y)    
            scores.append(mc_clone.n_captured.copy())
            
        return scores
        
    def calculate_score (self):
        
        scores = [sc[1] < sc[2] for sc in self.scores]   
        self.mu = np.mean(scores)
        self.sigma = np.std(scores)


    def depth_first_search_gp(self):

        embeddings = [self.game_embedding.copy()]
        mus = [self.mu]
        next_moves = self.possible_next_moves.copy()
        next_moves_ndx = [x for x in range(len(next_moves))]

        if len(self.children) > 0:
            for c in self.children:
                e, m, nm, nmx  = c.depth_first_search_gp()
                embeddings += e
                mus += m
                next_moves += nm
                next_moves_ndx += nmx
                
        return embeddings, mus, next_moves, next_moves_ndx

    
    def depth_first_search_move(self):

        best = self
        best_x = self.x
        best_y = self.y
        win_probability = self.mu
        
        if len(self.children) > 0:
            
            best, bx, by, win_probability = self.children[0].depth_first_search_move()
            best_x = self.children[0].x   # best x and y only come from direct descendents
            best_y = self.children[0].y   # not any lower
            
            for c in self.children[1:]:
                best_child, bx, by, wp = c.depth_first_search_move()
                if minimax_wp(best_child, best, self.game.current_player):
                    best = best_child
                    best_x = c.x
                    best_y = c.y
                    win_probability = wp
                    
        return best, best_x, best_y, win_probability
            

    def finished(self):
        return (self.game.n_moves - sum(self.game.n_captured)) == self.game.size**2





def parallel_acquisition(preds, covs, parallel):
    if parallel['n_parallel'] == 1:
        self.scores = [np.exp(m+s**2/2)*(1-norm.cdf(self.y_best, m+s**2, s)) for m in self.MUs]
    else:
#        url = 'https://boaz.onrender.com/qei?n={}&y_best={}'.format(parallel['n_parallel'], parallel['y_best'])
        url = 'http://localhost:8080/mpi'
        S_as_str = []
        for s in covs:
            S_as_str.append(';'.join([','.join([str(r) for r in row]) for row in s]))
            
#        return {'sigma': '|'.join(S_as_str),
#                'k': ';'.join([','.join([str(parallel['y_best']-x) for x in row]) for row in preds])}
        
        response = requests.post(url, data=json.dumps({'sigma': '|'.join(S_as_str),
                                                        'k': ';'.join([','.join([str(parallel['y_best']-x) for x in row]) for row in preds])}))
        try:
            jsponse = json.loads(response.content.decode('utf-8'))
            return [float(j) for j in jsponse['scores'].split(',')]
        except Exception as e:
            print(e)
            return [-1]
            


# same as MC method on Node
# except this can be 'mapped'
def get_best_batch(p):
    
    embeddings, scores, next_moves, next_move_ndx = p['root'].depth_first_search_gp()
    batches = []
    string_batches = []
    
    for x in range(p['n_gpr_samples']):
        batch = [next_moves[random.randint(0, len(next_moves)-1)] for y in range(p['n_parallel'])]
        batches.append(batch)
        string_batches.append(';'.join([','.join([str(x) for x in n.game_embedding]) for n in batch]))

    gpr_response = requests.post(url=p['gpr_url'], data=json.dumps({'batches': '|'.join(string_batches)}))
    
    mpi_response = requests.post(url=p['mpi_url'], data=gpr_response.content)
    try:
        jsponse = json.loads(mpi_response.content)
        mpi = [float(x) for x in jsponse['scores'].split(',')]
        mx = max(mpi)
        best_batch_id = mpi.index(mx)
    except Exception as e:
        return e

    return tuple([batches[best_batch_id], mpi])

def simulations(p):
    
    parent = p['move'].parent
    parent.children.append(Node(p['move'].game.copy(), p['move'].x, p['move'].y))
    parent.children[-1].scores = parent.children[-1].MC(p['n_simulations'])
    parent.children[-1].calculate_score()

    return 'Everything OK Mr. Jones'





















    
    scores = []
    for n in range(p['N']):
        first_move = True
        mc_clone = p['node'].game.copy()
        while (mc_clone.n_moves - np.sum(mc_clone.n_captured)) < mc_clone.size**2:
            x, y = mc_clone.get_random_move()
            mc_clone.place_stone(x, y)
            if first_move:
                    
                # Possible Next Moves are not MC'd
                    
                p['node'].possible_next_moves.append(Node(mc_clone.copy(), x, y))
                p['node'].possible_next_moves[-1].parent = p['node']
                first_move = False
                    
            if (mc_clone.n_moves - np.sum(mc_clone.n_captured)) < mc_clone.size**2:    
                x, y = mc_clone.get_random_move()
                mc_clone.place_stone(x, y)    
        scores.append(mc_clone.n_captured.copy())
            
    return (p['id'], scores)



    