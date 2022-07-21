import logging 
from gym.envs.registration import register  

logger = logging.getLogger(__name__)  

register(   
    id='TargetPathTrack-v0', 
    entry_point='srobot.envs:racetrack',    
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=8.0,
    nondeterministic = True, 
)

register(     
    id='PlantSim-v0',     
    entry_point='srobot.envs:python_env',  
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=10.0,    
    nondeterministic = True, 
)

register(     
    id='RobotEnv-v0',     
    entry_point='srobot.envs:SRobotEnv',    
#     timestep_limit=1000,
#     turns = 5,
    reward_threshold=1.0,    
    nondeterministic = True,
)


