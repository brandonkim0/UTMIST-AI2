# import skvideo
# import skvideo.io
from environment.environment import RenderMode
from environment.agent import SB3Agent, CameraResolution, RecurrentPPOAgent, BasedAgent, UserInputAgent, ConstantAgent, run_match, run_real_time_match
from user.train_agent import gen_reward_manager, get_latest_model
from user.my_agent import SubmittedAgent


reward_manager = gen_reward_manager()

experiment_dir = "checkpoints/experiment_0/" #input('Model experiment directory name (e.g. experiment_1): ')
model_name = "rl_model00_steps" #input('Name of first model (e.g. rl_model_100_steps): ')

my_agent = UserInputAgent()
#opponent = SubmittedAgent(None)
opponent = SubmittedAgent(get_latest_model('experiment_0'))
# my_agent = UserInputAgent()
# opponent = ConstantAgent()

num_matches = 2 #int(input('Number of matches: '))
#opponent=BasedAgent()
match_time = 50000000000
# 270
# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
