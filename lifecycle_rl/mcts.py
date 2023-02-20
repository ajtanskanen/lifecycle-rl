'''

Markov Chain Tree Search with Neural nets

as in Silver et al 2017

BUT:
- for single player
- not really for games
- to traverse and plan choices in a social security framework

'''

class TreeNode:
    def __init__(self, leaf=False,reward=-np.infty,infostate=None,state=None):
        self.reward = reward # cumulative reward
        self.state = state # state of the env
        self.infostate = infostate # if needed
        self.children = []
        self.leaf=leaf

class MCTS():
    def __init__(self,env,c):
        '''
        Init routines
        '''
        
        self.root = TreeNode()
        self.env = env
        self.C = c
        self.children=None
        
    def add_child(self,node,action):
        #self.env.set_state(node.state)
        state, reward, done, benq = self.env.step(action)
        newnode=TreeNode(leaf=True,reward=reward)
        node.leaf=False
        node.children.append(newnode)

    def search(self,node):
        if len(node.children)>0:
            a = self._selection(state)
    
    def _selection(self,):
        '''
        Phase 1: select best actions until the end is reached
        '''
    
        
            
    def _ucb(self,x_i,n_i):
        return x_i + self.C * math.sqrt(n(t)/n_i)
    
    def _expansion(self,):
    
    def _simulation(self,):
    
    def _backprop(self,):