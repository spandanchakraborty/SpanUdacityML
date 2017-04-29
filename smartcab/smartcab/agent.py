import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        ### Added to count the no# of trails till epsilon decays to Tolerance set for the simulator
        ### to switch to testing from Learning
        self.trialcounter=0


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon=0.0
            self.alpha=0.0
        else:
            #### Constant decay of epsilon by .05
            #self.epsilon=self.epsilon-0.001
            ### Exponential Decay of Epsilon with a=0.1
            ### Trail counter increases as the no# of trail increases resulting to exponential
            ### decay of epsilon
            ### self.epsilon=self.epsilon-.05
            self.epsilon=math.exp((-1)*.1*self.trialcounter)
            self.alpha=self.alpha*.999
            self.trialcounter=self.trialcounter+1

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent  
        ### Not using ,inputs['right'].. As per US Traffic rules when the light is Green for primary cab , the intended direction of the
        ### Cab at my Right doesnt matter for my next action.
        ### When the light is Red , only degree of freedom I have to go right when there is no Cab coming from Left in same lane. In this scenario
        ### as well my action is not dependent on the intended direction of the Cab to my right      

        ### Further from Review comments the traffic from Left is only relevant for learning when I have Red light and 
        ### the cab at left drives forward. All other cases traffic from left will not be of any concern
        ### hence further reducing the State space to inputs['left'] == 'forward'
        state = (waypoint, inputs['light'],inputs['oncoming'],inputs['left']=='forward')
        ### Not using ,inputs['right'].. As per US Traffic rules when the light is Green for primary cab , the intended direction of the
        ### Cab at my Right doesnt matter for my next action.
        ### When the light is Red , only degree of freedom I have to go right when there is no Cab coming from Left in same lane. In this scenario
        ### as well my action is not dependent on the intended direction of the Cab to my right

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        ### maxQ to hold the Action : Left /Right/Forward which has the Highest Q value for the state
        ### in an way the maxq is the policy we are after when a cab sees a state .. it takes action based
        ### on the highest Q value associated to the action

        ###maxQ = max(self.Q[state], key=self.Q[state].get)  # this returns the Key of the dictionary with Highest Value
                                                          # since the Actions & their Q value is a Dictionary returned by Q[State]
        maxQ=max(self.Q[state].values())  # gets the Max Value from the Action dictionary for the state

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if not state in self.Q:
            temp_act_dict={}
            for x in self.valid_actions:
                temp_act_dict[x]=0.0
            self.Q[state]=temp_act_dict

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if not self.learning:
            action=random.choice(self.valid_actions)
        else:
            ### epsilon probability aims to enable some random exploration early on in trails
            ### as epsilon decays the chance that the Cab will randomly explores will go down
            ### instead the Cab will follow the Q table for policy
            if self.epsilon>random.random():
                action=random.choice(self.valid_actions)
            else: 
                ### Get the action with the highest Q value from Q table for the State
                # in case of multiple actions in the state has same Q values == Max Q value, pick one of the actions randomly
                temp_max_Q_keys=[]
                for key1, val1 in self.Q[state].items():
                    if val1==self.get_maxQ(state):
                        temp_max_Q_keys.append(key1)
                if len(temp_max_Q_keys)==1:

                    action = max(self.Q[state], key=self.Q[state].get)  # Get the Action that has max q val
                    #### print('Only 1 key/action:', action)
                else:
                    action=random.choice(temp_max_Q_keys)
                    ### print('More than 1 key/action found:', self.Q[state],'//Action Taken:', action)
 
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        #### self.Q[state][action]=(1-self.alpha)*self.Q[state][action]+reward*self.alpha
        self.Q[state][action] += self.alpha*(reward - self.Q[state][action])

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True,alpha=1)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=.01,display=False,log_metrics=True,optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=20,tolerance=.00000001)


if __name__ == '__main__':
    run()
