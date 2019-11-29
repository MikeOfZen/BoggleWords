import numpy as np
import random
import collections

words_file="words_alpha.txt"
ALL_LETTERS='qwertyuiopasdfghjklzxcvbnm'
allowed_moves=np.array([
(0,0), #stay in place, filtered by ALLOW_OUROBOROS setting
(+1,0),  #0deg
(0,+1),  #90
(-1,0),  #180
(0,-1),  #270
(+1,+1), #45
(-1,+1), #135
(-1,-1), #225
(+1,-1), #315
])


FORBID_OUROBOROS=True

class WordsTree():
    def __init__(self):
        self.root = Node("")

    def add_word(self,word: str) -> None:
        current_node=self.root
        for letter in word:
            current_node=current_node.get_create_node(letter)
        if current_node.is_word:
            raise ValueError("Word already exists")
        current_node.is_word=True

    def check_word(self,word:str)->bool:
        current_node=self.root
        for letter in word:
            try:
                current_node=current_node[letter]
            except KeyError:
                return False
        return current_node.is_word

class Node(collections.MutableMapping):
    def __init__(self,letter,is_word=False):
        self.letter=letter
        self.is_word=is_word
        self.store = dict()

    def __str__(self):
        return self.letter

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key.lower()

    def get_create_node(self, letter: str):
        try:
            return self[letter]
        except KeyError:
            self[letter]=Node(letter)
        return self[letter]

def random_char():
    while True:
        yield random.choice(ALL_LETTERS)

class BoggleBoard(np.ndarray):
    def __new__(cls,shape:np.ndarray,source=iter(random_char())):
        shape=np.array(shape,dtype=np.int)
        one_dim=np.fromiter(source,count=shape.prod(),dtype=np.dtype((str,1)))
        return one_dim.reshape(shape)


class Miners():
    def __init__(self,board,word_tree):
        self.board=board
        self.results=[]
        self.word_tree=word_tree

    def init_word_miner(self,position:np.ndarray):
        position = np.array(position)
        assert self._check_positions(position)
        #position = np.expand_dims(position,axis=0)
        return self._word_miner(position, "", np.zeros((0, 2)), self.word_tree.root)

    def _check_positions(self, pos):
        if min(pos)>=0 and max(pos-self.board.shape)<0:
            return True
        return False

    def _word_miner(self, position:np.ndarray, string_tail:str, positions_tail:np.ndarray, previous_node:Node)->None:
        current_letter=self.board[tuple(position)]
        string_so_far=string_tail+current_letter

        positions_so_far=np.concatenate((positions_tail,np.expand_dims(position,axis=0)))

        try:
            current_node=previous_node[current_letter]
        except KeyError:    #this means that there is no such word
            return

        if current_node.is_word:
            word=string_so_far
            self.results.append((word,positions_so_far))

        possible_moves= position + allowed_moves
        legal_moves=np.array(list(filter(self._check_positions, possible_moves)))
        if FORBID_OUROBOROS:
            legal_moves=np.array(list(filter(lambda move:not (move ==positions_so_far).all(axis=-1).any(),legal_moves)))
        for move in legal_moves:
            self._word_miner(move, string_so_far, positions_so_far, current_node)
        return

def get_grid_coords(shape):
    dim_ranges=[np.arange(dim_size) for dim_size in shape]
    ddims=np.meshgrid(*dim_ranges)
    grid=np.stack(ddims,axis=-1)
    grid_flat=grid.reshape((-1,len(shape)))
    return grid_flat

def main(shape=(100,100)):
    print("reading words list")
    tree=WordsTree()
    with open(words_file,"r") as f:
        for line in f:
            line=line.rstrip()
            tree.add_word(line)

    board = BoggleBoard(np.array(shape))
    miners = Miners(board, tree)

    coords_list=get_grid_coords(shape)

    print("Board:")
    for x in board:
        for y in x:
            print(y, end="")
        print()


    for i,coord in enumerate(coords_list):  #naive loop implementation, can be improved by threading
        miners.init_word_miner(coord)
        if not i%shape[0]:
            print(".",end="",flush=True)

    print("Found %d words" % len(miners.results))
    for result in miners.results:
        print(result[0])

if __name__=="__main__":
    main()