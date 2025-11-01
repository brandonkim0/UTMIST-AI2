# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):

    def __init__(
            self, hold_frames: int = 1,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.hold_frames = max(1, int(hold_frames))
        self.jumping = False
        self.bufferJump = 0
        self.bufferPickup = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [11]
        plr_KO = self.obs_helper.get_section(obs, 'player_state') in [11]
        action = self.act_helper.zeros()
        plrWeapon = self.obs_helper.get_section(obs, 'player_weapon_type')
        plrFacing = self.obs_helper.get_section(obs, 'player_facing')
        spawners = self.obs_helper.get_section(obs, 'player_spawner_1')
        jumps = self.obs_helper.get_section(obs, 'player_jumps_left')
        inGap = False
        #print(spawners)
        #print(jumps)
        
        if plr_KO:
            return action
        
        if 2 > pos[0] > -2:
            inGap = True
            if jumps == [0.]:
                action = self.act_helper.press_keys(['a'])
                return action
                

        if self.jumping:
            if opp_KO:
                self.jumping = False
                return action
            if self.obs_helper.get_section(obs, 'player_state') == [1.]:
                action = self.act_helper.press_keys(['space','s','j'], action)
                self.jumping = False
            return action

        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        if plrWeapon != [2.]:
            if spawners[2] == [1.]:
                if plr_KO:
                    return action
                if pos[0] < spawners[0]:
                    action = self.act_helper.press_keys(['d'])
                elif pos[0] > spawners[0]:
                    action = self.act_helper.press_keys(['a'])
                if pos[1] > spawners[1]:
                    action = self.act_helper.press_keys(['space'], action)
                if (abs(pos[0] - spawners[0] < 0.9)) and abs(pos[1] - spawners[1]) < 0.7:
                    if self.bufferPickup == 0:
                        action = self.act_helper.press_keys(['h'], action)
                        self.bufferPickup = 1
                    else:
                        self.bufferPickup = 0
            elif not inGap:
                if not opp_KO:
                    if (opp_pos[0] > pos[0]):
                        action = self.act_helper.press_keys(['d'])
                    else:
                        action = self.act_helper.press_keys(['a'])
                if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
                    action = self.act_helper.press_keys(['space'], action)
                if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
                    action = self.act_helper.press_keys(['j'], action)
                return action
        else:
            if pos[0] < 4 or pos[0] > 7:
               if pos[0] < 6:
                   action = self.act_helper.press_keys(['d'], action)
               else:
                   action = self.act_helper.press_keys(['a'], action)
            else:
                if opp_KO:
                    return action 
                if opp_pos[0] - pos[0] > -0.2:
                    if plrFacing == [0.]:
                        action = self.act_helper.press_keys(['d'], action)
                    action = self.act_helper.press_keys(['s','j'], action)
                if -0.2 <= (pos[0] - opp_pos[0]) <= 0.6:
                    if (pos[1] - opp_pos[1]) > 0.1:
                        action = self.act_helper.press_keys(['space','s','j'], action)
                    else:
                        if plrFacing == [1.]:
                            action = self.act_helper.press_keys(['a'], action)
                        action = self.act_helper.press_keys(['s','j'], action)
                        self.jumping = True
            if pos[1] > 0.85  and self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
        return action