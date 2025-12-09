
from openai import OpenAI
import os 
import time
import ast
import json
import numpy as np
# import pose_mask
import torch
import sys
import traceback
# from llm.assertions import *
if __name__ != "main":
    sys.path.append("/source/inyup/IEFA/src/FaceMEO/")
from llm.motion import FacialMotion
from llm.motion_io import Motion_DB

client = OpenAI(
    api_key="" # VML public key
)

def initialize_preprompts(initial_prompt_sequence):
    # read_progprompt_0(initial_prompt_sequence)
    # read_progprompt("",initial_prompt_sequence)
    get_incontext(initial_prompt_sequence)
    return initial_prompt_sequence
    
def first_prompt_sequence(initial_prompt_sequence, conversation_history):
    return initial_prompt_sequence + conversation_history 

messages = []

def query_gpt():
    MODEL = "gpt-4o"
    # MODEL = "gpt-3.5-turbo"
    # import pdb;pdb.set_trace()
    response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0)
    content = response.choices[0].message.content
    return content

def sequence_content(content, prompt2, append=True):
    if(append):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": prompt2})
    else:
        messages[-2] = {"role": "assistant", "content": content}
        messages[-1] = {"role": "user", "content": prompt2}

def query_gpt_sequential(prompt_sequence):
        responses = []
        # import pdb;pdb.set_trace()
        for i in range(0, len(prompt_sequence) - 1, 2):
            messages.append({"role": "user", "content": prompt_sequence[i]})
            messages.append({"role": "assistant", "content": prompt_sequence[i + 1]})
        messages.append({"role": "user", "content": prompt_sequence[-1]})
        content = query_gpt()
        responses.append(content)
        messages.clear()
        # for i in range(len(prompt_sequence)):
        #     messages.append({"role" : "user", "content" : prompt_sequence[i]})
        # content = query_gpt()
        # responses.append(content)
        # messages.clear()
        return responses[-1]

def old_v1_response_to_code(responses, err_prompt_sequence, try_counter, logger = None, trylist=[], context = {}):
    """
    This function processes the responses from the GPT model, 
    attempts to execute code snippets found in the responses, 
    and handles errors if they occur. 
    It also manages retries up to a certain limit (try_counter)
    """
    if try_counter >= 3:
        print("Giving up")
        return None, -1
    responses_split = responses.split("\n")
    methods = []
    valid = True
    not_defined = False
    found_load_motion = False
    found_save_motion = False
    # print(responses) # code created 
    counter = 0

    ## TODO ##
    ## - (DONE) change this to retrieve face version
    for response in responses_split:
        # if "do_" in response and "(" in response and "undo" not in response:
        #     methods.append(response.strip())
        if "set_" in response or "activate_" in response or "FacialMotion(" in response:
            methods.append(response.strip())
        elif " = [" in response or "= \" " in response:
            methods.append(response.strip())
        elif " = " in response:
            methods.append(response.strip())
        elif "load_motion" in response:
            methods.append(response.strip())
            found_load_motion = True
        elif "save_motion" in response:
            methods.append(response.strip())
            found_save_motion = True
    
    ## this is the part where created code is actually executed for test -> assertions need to be set(completed)
    for method in methods:
        # import pdb;pdb.set_trace()
        try:
            # print(method)
            # success, err = eval(method) # attempts to execute the method using Python's eval() function, which evaluates a string as Python code
            exec(method, globals(), context) # assignment operation only works with 'exec' and globals() needed to let exec() know current imported modules, and context for retrieving exec()'s retrived global variables
            # if success < 0: # means it's error
            #     err_prompt_sequence.append(responses)
            #     tb = traceback.format_exc() # logs the traceback (tb)
            #     err_prompt_sequence.append(err)
            #     valid = False
            #     break
        except Exception as err: # catches any exceptions raised during the execution of eval()
            print("try except", err) 
            print(responses)
            err_prompt_sequence.append(responses)
            tb = traceback.format_exc()
            err_prompt_sequence.append(str(err))
            valid = False
            break
    ## checks any methods were found or if both load_motion and save_motion were detected
    found_methods = False
    if len(methods) > 0: 
        found_methods = True
    elif len(methods) == 0 and (found_load_motion or found_save_motion):
        found_methods = True
    else:
        print(methods) # likely empty, b/c no methods

    if not found_methods and not not_defined :
        print("Invalid Program")
        # print(responses)
        err_prompt_sequence.append(responses)
        err_prompt_sequence.append("Please respond by editing your invalid program.") # asking the GPT model to edit the invalid program.
        valid = False
    trylist.append(counter) # trylist keeps track of the attempts
    if not valid: # If the response was not valid
        # if logger:
        #     logger.log_error(responses)
        code, responses, context = query_model(err_prompt_sequence, err_prompt_sequence, try_counter + 1, logger, trylist, context) # to retry with an updated err_prompt_sequence, incrementing try_counter by 1.
        counter += 1
        #trylist.append(counter)
    else: # # If the response was valid, it assigns the responses to code.
        print("Code successfully executed without error!!")
        code = responses

    return code, counter, context

def old_v2_response_to_code(responses, err_prompt_sequence, try_counter, logger=None, trylist=[], context={}):
    """
    - process the responses from GPT
    - attempts to execute code snippets found in the responses
    - handles errors
    - manages retries up to a limit(try_counter)
    """
    max_retries = 3
    
    while try_counter < max_retries:
        try:
            response_split = responses.split('\n')
            methods = []
            valid = True
            found_load_motion = False
            found_save_motion = False
            counter = 0

            # parse the responses and extract methods
            for response in response_split:
                if ("Motion_DB(" in response) or ("direct_" in response) or ("set_" in response) or ("activate_" in response) or ("FacialMotion(" in response):
                    methods.append(response.strip())
                elif (" = [" in response) or ("= \" " in response):
                    methods.append(response.strip())
                elif " = " in response:
                    methods.append(response.strip())
                elif "load_motion" in response:
                    methods.append(response.strip())
                    found_load_motion = True
                elif "save_motion" in response:
                    methods.append(response.strip())
                    found_save_motion = True
            
            for method in methods:
                try:
                    exec(method, globals(), context) # Execute the generated code
                except Exception as err:
                    print("Error while executing method: ", err)
                    err_prompt_sequence.append(responses)
                    tb = traceback.format_exc()
                    err_prompt_sequence.append(str(err))
                    valid = False
                    break
            
            found_methods = len(methods) > 0 or found_load_motion or found_save_motion
            
            if not found_methods:
                print("Invalid Program")
                err_prompt_sequence.append(responses)
                err_prompt_sequence.append("Please respond by editing your invalid program.")
                valid =False
            
            trylist.append(counter)
            if valid:
                print("Code successfully executed without error!")
                return responses, counter, context
        
        
        except Exception as main_err:
            print(f"Main Error:", main_err)
            err_prompt_sequence.append(str(main_err))
        
        # increase retry counter
        try_counter += 1
        if try_counter < max_retries:
            # retry with updated err_prompt_seqence
            print(f"Retrying... Attempt {try_counter}")
            responses = query_gpt_sequential(err_prompt_sequence)
    
    print("Giving up after maximu retries...")
    return None, -1, context

def response_to_code(responses, err_prompt_sequence, try_counter, logger=None, trylist=[], context={}):
    """
    - process the responses from GPT
    - attempts to execute code snippets found in the responses
    - handles errors
    - manages retries up to a limit(try_counter)
    """
    max_retries = 4
    # codes = []
    while try_counter < max_retries:
        try:
            response_split = responses.split('\n')
            methods = []
            valid = True
            found_load_motion = False
            found_save_motion = False
            counter = 0

            # parse the responses and extract methods
            for response in response_split:
                if ("Motion_DB(" in response) or ("FacialMotion(" in response):
                    methods.append(response.strip())
                elif ("generate_keyframe(" in response) or ("undo_last_motion(" in response) or ("revert_to_motion(" in response):
                    methods.append(response.strip())
                elif (" = [" in response) or ("= \" " in response):
                    methods.append(response.strip())
                elif " = " in response:
                    methods.append(response.strip())
                elif "db.load_motion" in response:
                    methods.append(response.strip())
                    found_load_motion = True
                elif "db.save_motion" in response:
                    methods.append(response.strip())
                    found_save_motion = True
            # import pdb;pdb.set_trace()
            for method in methods:
                # codes.append(method)
                try:
                    exec(method, globals(), context) # Execute the generated code
                except Exception as err:    
                    print("Error while executing method: ", err)
                    err_prompt_sequence.append(responses)
                    tb = traceback.format_exc()
                    err_prompt_sequence.append(str(err))
                    valid = False
                    break
            
            found_methods = len(methods) > 0 or found_load_motion or found_save_motion
            
            if not found_methods:
                print("Invalid Program")
                err_prompt_sequence.append(responses)
                err_prompt_sequence.append("Please respond by editing your invalid program.")
                valid =False
            
            trylist.append(counter)
            if valid:
                # codes.append(responses)
                print("Code successfully executed without error!")
                return responses, counter, context
        
        except Exception as main_err:
            print(f"Main Error:", main_err)
            err_prompt_sequence.append(str(main_err))
        
        # increase retry counter
        try_counter += 1
        if try_counter < max_retries:
            # retry with updated err_prompt_seqence
            print(f"Retrying... Attempt {try_counter}")
            responses = query_gpt_sequential(err_prompt_sequence)
    
    print("Giving up after maximum retries...")
    return None, -1, context


def query_model(prompt, err_prompt_sequence, try_counter, logger = None, trylist=[], context = {}):
    print("querying model")
    start_time = time.time()
    responses = query_gpt_sequential(prompt) # get code from GPT
    code , responses, context = response_to_code(responses, err_prompt_sequence, try_counter, logger, trylist, context)
    end_time = time.time()
    print(f"took {(end_time - start_time):6f} seconds for current edit...")

    return code, responses, context

def read_progprompt(edit_instruction, prompt_sequence):
    print(edit_instruction)
    if __name__ == "__main__":
        path = "llm/prog_prompt3.py"
    else: # case when run from run_IEFA.py
        path = "src/FaceMEO/llm/prog_prompt3.py"
    with open(path, "r") as f:
        lines = f.read()
        prompt_sequence.append("```python\n" + lines + "```")
    return prompt_sequence

def read_progprompt_0(prompt_sequence):
    if __name__ == "__main__":
        path = "llm/motion.py"
    else: # case when run from run_IEFA.py
        path = "src/FaceMEO/llm/motion.py"
    with open(path, "r") as f:
            lines = f.read()
            prompt_sequence.append("```python\n" + lines + "```")
    return prompt_sequence

def get_incontext(prompt_sequence):
    
    prompt_sequence.append('''
            ### You are editing a facial animation sequence using the MotionAPI defined in the FacialMotion class, and managing history with Motion_DB.

            # -----------------------
            # Motion API Overview
            # -----------------------

            # Create and manage motions
                db = Motion_DB()
                motion = FacialMotion()                            # Start from scratch
                loaded = db.load_motion("motion_n")                # Load a saved motion
                motion = FacialMotion(loaded)                      # Initialize from loaded motion

            # Editing expressions
                generate_keyframe(
                    frame=int,  # single int value to insert keyframe
                    identifiers=list,  # ["emotion", "FACS", "blendshape"]
                    intensity=list,    # match the number of values in this list to the number of elements in 'identifiers'
                    delta_mode=True/False
                )
                    - frame:
                        • A single integer value specifying the frame at which to insert or modify a keyframe.
                        • Only one frame per generate_keyframe() call is allowed.
                        • If you want to insert multiple frames, call generate_keyframe() separately for each frame.

                    - identifiers list can have:
                        • High-level emotion presets
                            ### Emotion-level identifiers (high-level presets):
                            'angry'      → [brow_lowerer, outer_brow_raiser, lid_tightener, nose_wrinkler, lip_stretcher, chin_raiser, jaw_thrust, lip_presser]  
                            'contempt'   → [dimpler, lip_corner_puller, nose_wrinkler, lip_presser, brow_lowerer, inner_brow_raiser, lip_presser, chin_raiser]  
                            'disgusted'  → [nose_wrinkler, upper_lip_raiser, lip_corner_depressor, jaw_thrust]  
                            'fear'       → [inner_brow_raiser, outer_brow_raiser, upper_lid_raiser, lid_tightener, mouth_stretch, jaw_sideways_left, jaw_sideways_right, chin_raiser]  
                            'happy'      → [lip_corner_puller, cheek_raiser, lid_tightener, upper_lid_raiser, inner_brow_raiser, outer_brow_raiser]  
                            'sad'        → [inner_brow_raiser, brow_lowerer, lip_corner_depressor, chin_raiser, lower_lip_depressor, blink]  
                            'surprised'  → [inner_brow_raiser, outer_brow_raiser, upper_lid_raiser, mouth_stretch, jaw_thrust, lip_funneler]
                        • Mid-level FACS units 
                            ### FACS unit identifiers (mid-level expressions):
                                'inner_brow_raiser': ['brow_inner_up_l', 'brow_inner_up_r'],            
                                'outer_brow_raiser': ['brow_outer_up_l', 'brow_outer_up_r'],            
                                'brow_lowerer': ['brow_down_l', 'brow_down_r'],                         
                                'cheek_raiser': ['cheek_squint_l', 'cheek_squint_r'],                   
                                'upper_lid_raiser': ['eye_wide_l', 'eye_wide_r'],                       
                                'lid_tightener': ['eye_squint_l', 'eye_squint_r'],                      
                                'nose_wrinkler': ['nose_sneer_l', 'nose_sneer_r'],                      
                                'upper_lip_raiser': ['mouth_shrug_upper'],                              
                                'nasolabial_deepener': ['mouth_upper_up_l', 'mouth_upper_up_r'],        
                                'lip_corner_puller': ['mouth_smile_l', 'mouth_smile_r'],                
                                'dimpler': ['mouth_dimple_l', 'mouth_dimple_r'],                        
                                'lip_corner_depressor': ['mouth_frown_l', 'mouth_frown_r'],             
                                'lower_lip_depressor': ['mouth_lower_down_l', 'mouth_lower_down_r'],     
                                'chin_raiser': ['mouth_shrug_lower'],                                    
                                'lip_pucker': ['mouth_pucker'],                                          
                                'lip_stretcher': ['mouth_stretch_l', 'mouth_stretch_r'],                 
                                'lip_funneler': ['mouth_funnel'],                                        
                                'lip_presser': ['mouth_press_l', 'mouth_press_r'],                       
                                'mouth_stretch': ['jaw_open'],                                           
                                'lip_suck': ['mouth_roll_lower', 'mouth_roll_upper'],                    
                                'jaw_thrust': ['jaw_forward'],                                          
                                'jaw_sideways_left': ['jaw_left'],                                      
                                'jaw_sideways_right': ['jaw_right'],                                     
                                'cheek_blow': ['cheek_puff_l', 'cheek_puff_r'],                          
                                'blink': ['eye_blink_l', 'eye_blink_r']
                        • Low-level blendshape targets 
                            ### Direct blendshape identifiers (low-level):
                                'brow_down_l', 'brow_down_r', 'brow_inner_up_l', 'brow_inner_up_r', 'brow_outer_up_l', 'brow_outer_up_r',
                                'cheek_puff_l', 'cheek_puff_r', 'cheek_squint_l', 'cheek_squint_r',
                                'eye_blink_l', 'eye_blink_r', 'eye_look_down_l', 'eye_look_down_r',
                                'eye_look_in_l', 'eye_look_in_r', 'eye_look_out_l', 'eye_look_out_r',
                                'eye_look_up_l', 'eye_look_up_r', 'eye_squint_l', 'eye_squint_r',
                                'eye_wide_l', 'eye_wide_r', 'jaw_forward', 'jaw_left', 'jaw_open', 'jaw_right',
                                'mouth_close', 'mouth_dimple_l', 'mouth_dimple_r', 'mouth_frown_l', 'mouth_frown_r',
                                'mouth_funnel', 'mouth_left', 'mouth_lower_down_l', 'mouth_lower_down_r',
                                'mouth_press_l', 'mouth_press_r', 'mouth_pucker', 'mouth_right',
                                'mouth_roll_lower', 'mouth_roll_upper', 'mouth_shrug_lower', 'mouth_shrug_upper',
                                'mouth_smile_l', 'mouth_smile_r', 'mouth_stretch_l', 'mouth_stretch_r',
                                'mouth_upper_up_l', 'mouth_upper_up_r', 'nose_sneer_l', 'nose_sneer_r'
                    - intensity:
                        • should be a list of Scalar values for all specified elements in 'identifiers'.
                        • List for specifying per-identifier intensity.
                    - delta_mode:
                        • False → add a new keyframe
                        • True → modify existing keyframe by adding/subtracting values

            # Finalizing
                db.save_motion(motion, "motion_n")
            
            # -----------------------
            # Editing Strategy Guide
            # -----------------------
            
            # [1] ADD (New Expression)
                - Instruction keywords: "add", "insert", "make", "apply", "activate"
                - Action:
                    - motion.generate_keyframe(
                        frame=..., 
                        identifiers=[...], 
                        intensity=[...], 
                        delta_mode=False
                    )

            # [2] MODIFY (Adjust Existing Expression)
                - Instruction keywords: "edit", "adjust", "boost", "reduce", "enhance", "weaken"
                - Action:
                    - motion.generate_keyframe(
                        frame=..., 
                        identifiers=[...], 
                        intensity=[...], 
                        delta_mode=True
                    )

            # [3] UNDO (Restore Previous Motion)
                - Instruction keywords: "undo", "revert to previous", "go back to previous"
                - Action:
                    - loaded = db.undo_last_motion()
                    - motion = FacialMotion(loaded)
                    - db.save_motion(motion, "motion_n")

            # [4] REVERT TO SPECIFIC SNAPSHOT
                - Instruction keywords: "revert to [X] expression", "restore to before [X]", "go back to motion_n-i"
                - Search all previous edits and find & detect where that [X] expression is and load that motion
                - Simply load that `motion_n-i`, if indicated directly
                - Action:
                    - loaded = db.revert_to_motion("motion_n-i")
                    - motion = FacialMotion(loaded)
                    - db.save_motion(motion, "motion_n")

            # [5] CLEAR (Reset to Neutral)
                - Instruction keywords: "clear all", "reset"
                - Action:
                    - motion = FacialMotion()
                    - db.save_motion(motion, "motion_n")

            # -----------------------
            # Final Rules
            # -----------------------

            - Use only the provided API methods exactly as defined. Do not define any new classes or functions.
            
            - Initialize `db = Motion_DB()` once, just before the **first edit**.
            
            - At the **first edit**, always initialize an empty motion using `motion = FacialMotion()`. Never load any prior motion.
            
            - For [1], [2] instructions:
                - Always load the motion saved in **the previous edit** using:
                    - `loaded = db.load_motion("motion_n-1")`
                    - `motion = FacialMotion(loaded)`

            - For [3] instructions: 
                - Always load the same one loaded in **the previous edit** like:
                    - `loaded = db.load_motion("motion_n-2")`
                    - `motion = FacialMotion(loaded)` 
                
            - For [4] instructions:
                - Revert to a indicated past edit **motion_k**, then re-initialize the motion with that edit:
                    - `loaded = db.revert_to_motion("motion_k")`
                    - `motion = FacialMotion(loaded)`
                    
            - For [5] insturctions: 
                - Never load any motion, just reinitialize with empty FacialMotion:
                    - `motion = FacialMotion()`

            - All edits must be saved using `db.save_motion(motion, "motion_n")`, where `n` is the current edit count:
                - `motion_1` for the first edit, `motion_2` for the second edit, and so on.
                - This numbering must increase **monotonically** regardless of whether edits are [1], [2], [3], [4] and[5].

            - Always include **inline comments** that explain:
                - Why each API call is made,
                - How the user instruction maps to the API call and arguments.

            - Keep function bodies minimal, correct, and directly reflecting the instruction intent.
            
            - Assume 30 fps: "1 second" = 30 frames.
            
            - If edit specifies certain length of frames, when first initializing FacialMotion() or Clear back to initial motion. 
                - 'motion = FacialMotion(total_frames = {specified_frame_length})'
                - ex) initialize with sequence length 126 -> 'motion = FacialMotion(total_frames=126)'
                
            - If not length of frames specified, 
                - 'motion = FacialMotion()'
    ''')

    prompt_sequence.append('''
        Now you will be given a series of user instructions paired with python functions.
        
        Each example includes:
            - A user instruction (as a comment)
            - The corresponding Python function that edits facial motion using defined APIs.
            
        Starting from motion_1 to motion_n (sequential numbering).
        
        Now study carefully and strictly follow the patterns in the following given simulated examples.
    ''')

    prompt_sequence.append("# Add a happy expression at frame 20 and return to neutral at frame 60. The frame length is 45\n")
    prompt_sequence.append('''
    db = Motion_DB()
    def add_happy_neutral():
        # It's first edit, no loading
        # Total frame length is 45, so initialize FacialMotion with 'total_frames=45'
        motion = FacialMotion(total_frames=45)

        # "Add a happy expression at frame 20"
        motion.generate_keyframe(frame=20, identifiers=["happy"], intensity=[1.0], delta_mode=False)

        # "Return to neutral at frame 60"
        motion.generate_keyframe(frame=60, identifiers=[], intensity=[], delta_mode=False)
        
        # This is 1st edit so "motion_1"
        db.save_motion(motion, "motion_1")
    ''')

    prompt_sequence.append("# At frame 30, enhance the left side of the smile slightly.\n")
    prompt_sequence.append('''
    def boost_left_smile():
        # It's [2], so just load the previous edit
        loaded = db.load_motion("motion_1")
        motion = FacialMotion(loaded)

        # "Enhance the left side of the smile slightly at frame 30"
        motion.generate_keyframe(frame=30, identifiers=["mouth_smile_l"], intensity=[0.2], delta_mode=True)

        # This is 2nd edit so "motion_2"
        db.save_motion(motion, "motion_2")
    ''')

    prompt_sequence.append("# Undo.\n")
    prompt_sequence.append('''
    def undo():
        # it's [3], just call `undo_last_motion()`
        loaded = db.undo_last_motion()
        motion = FacialMotion(loaded)

        # "Enhance the left side of the smile slightly at frame 30"
        motion.generate_keyframe(frame=30, identifiers=["mouth_smile_l"], intensity=[0.2], delta_mode=True)

        # This is 3rd edit so "motion_3"
        db.save_motion(motion, "motion_3")
    ''')

    prompt_sequence.append("# Insert a surprised reaction at frame 45 and fade it out by frame 80.\n")
    prompt_sequence.append('''
    def surprise_and_fade():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_3")
        motion = FacialMotion(loaded)

        # "Insert a surprised reaction at frame 45"
        motion.generate_keyframe(frame=45, identifiers=["surprised"], intensity=[1.0], delta_mode=False)

        # "Fade it out by frame 80"
        motion.generate_keyframe(frame=80, identifiers=[], intensity=[], delta_mode=False)

        # This is 4th edit so "motion_4"
        db.save_motion(motion, "motion_4")
    ''')

    prompt_sequence.append("# Boost the brow raise at frame 45 to make the surprise stronger.\n")
    prompt_sequence.append('''
    def boost_brow_raise():
        # it's [2], so just load the previous edit
        loaded = db.load_motion("motion_4")
        motion = FacialMotion(loaded)

        # "Boost the brow raise at frame 45"
        motion.generate_keyframe(frame=45, identifiers=["brow_outer_up_l", "brow_outer_up_r"], intensity=[0.5,0.5], delta_mode=True)
        
        # 5th edit so "motion_5"
        db.save_motion(motion, "motion_5")
    ''')

    prompt_sequence.append("# Add a soft blink at frame 35 and open eyes again at frame 50.\n")
    prompt_sequence.append('''
    def soft_blink():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_5")
        motion = FacialMotion(loaded)

        # "Add a soft blink at frame 35"
        motion.generate_keyframe(frame=35, identifiers=["blink"], intensity=[1.0], delta_mode=False)

        # "Open eyes again at frame 50"
        motion.generate_keyframe(frame=50, identifiers=["blink"], intensity=[-1.0], delta_mode=True)
        
        # 6th edit so "motion_6"
        db.save_motion(motion, "motion_6")
    ''')

    prompt_sequence.append("# Create a big jaw open expression at frame 25 and return to neutral at frame 70.\n")
    prompt_sequence.append('''
    def big_jaw_open():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_6")
        motion = FacialMotion(loaded)

        # "Create a big jaw open at frame 25"
        motion.generate_keyframe(frame=25, identifiers=["jaw_thrust"], intensity=[1.5], delta_mode=False)

        # "Return to neutral at frame 70"
        motion.generate_keyframe(frame=70, identifiers=[], intensity=[], delta_mode=False)
        
        # 7th edit so "motion_7"
        db.save_motion(motion, "motion_7")
    ''')

    prompt_sequence.append("# Slightly boost cheek raise while smiling at frame 25.\n")
    prompt_sequence.append('''
    def boost_cheek_raise():
        # it's [2], so just load the previous edit
        loaded = db.load_motion("motion_7")
        motion = FacialMotion(loaded)

        # "Slightly boost cheek raise at frame 25"
        motion.generate_keyframe(frame=25, identifiers=["cheek_raiser"], intensity=[0.2], delta_mode=True)
        
        # 8th edit so "motion_8"
        db.save_motion(motion, "motion_8")
    ''')

    prompt_sequence.append("# Introduce mild disgust with nose wrinkle at frame 55.\n")
    prompt_sequence.append('''
    def mild_disgust():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_8") 
        motion = FacialMotion(loaded) 

        # "Introduce mild disgust at frame 55"
        motion.generate_keyframe(frame=55, identifiers=["disgusted"], intensity=[0.7], delta_mode=False)

        # "Add nose wrinkle at frame 55 to emphasize disgust"
        motion.generate_keyframe(frame=55, identifiers=["nose_wrinkler"], intensity=[0.2], delta_mode=True)
        
        # 9th edit so "motion_9"
        db.save_motion(motion, "motion_9") 
    ''')

    prompt_sequence.append("# Clear all.\n") 
    prompt_sequence.append('''
    def mild_disgust():
        # it's [5], so don't load anything
        motion = FacialMotion()
        
        # 10th edit so "motion_10"
        db.save_motion(motion, "motion_10")
    ''')

    prompt_sequence.append("# Add a sad expression at frame 40 and neutralize at frame 90.\n")
    prompt_sequence.append('''
    def sad_expression():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_10")
        motion = FacialMotion(loaded)

        # "Add a sad expression at frame 40"
        motion.generate_keyframe(frame=40, identifiers=["sad"], intensity=[1.0], delta_mode=False)

        # "Neutralize at frame 90"
        motion.generate_keyframe(frame=90, identifiers=[], intensity=[], delta_mode=False)

        # 11th edit so "motion_11"
        db.save_motion(motion, "motion_11")
    ''')

    prompt_sequence.append("# Revert to the one added surprised expression.\n")
    prompt_sequence.append('''
    def sad_expression():
        # it's [4], and 'surprised' expression was added at 4th edit and saved as "motion_4" so load motion_4
        loaded = db.revert_to_motion("motion_4")
        motion = FacialMotion(loaded)
        
        # 12th edit so "motion_12"
        db.save_motion(motion, "motion_12") 
    ''')

    prompt_sequence.append("# Clear all again.\n") 
    prompt_sequence.append('''
    def clear_all():
        # it's [5], so don't load anything
        motion = FacialMotion()
        
        # 13th edit so "motion_13"
        db.save_motion(motion, "motion_13")
    ''')

    prompt_sequence.append("# Add a surprised expression approximately halfway through the sequence, then switch to an angry expression close to the end.\n")
    prompt_sequence.append('''
    def surprised_to_angry():
        # it's [1], so just load the previous edit
        loaded = db.load_motion("motion_13")
        motion = FacialMotion(loaded)
        
        # "Approximately halfway through the sequence" → assume frame 75 out of 150
        motion.generate_keyframe(frame=75, identifiers=["surprised"], intensity=[1.0], delta_mode=False)

        # "Close to the end" → interpreted as frame 140
        motion.generate_keyframe(frame=140, identifiers=["angry"], intensity=[1.0], delta_mode=False)

        # 14th edit so "motion_14"
        db.save_motion(motion, "motion_14")
    ''')

    prompt_sequence.append("# Adjsut the previous edit, instead of switching to angry near the end, transition to angry earlier — roughly midway between the midpoint and the end.\n")
    prompt_sequence.append('''
    def shift_angry_earlier():
        # It seems like [2] but it is not, since we are modifying the previous behaviour itself. 
        # So undo it and then redo the previous edit with slightly changed function call. 
        # It's [3] and then [1], call `undo_last_motion()` and then call the same functions with changed arguments. 
        loaded = db.undo_last_motion()
        motion = FacialMotion(loaded)

        # Reapply the surprised expression at the midpoint
        motion.generate_keyframe(frame=75, identifiers=["surprised"], intensity=[1.0], delta_mode=False)

        # "Roughly midway between midpoint (75) and end (140)" → interpret as frame 107
        motion.generate_keyframe(frame=107, identifiers=["angry"], intensity=[1.0], delta_mode=False)

        # 15th edit so "motion_15"
        db.save_motion(motion, "motion_15")
    ''')

    prompt_sequence.append('''
        Always save each edited motion sequentially:
        - Save the first edit as "motion_1", the next as "motion_2", and so on.
        - The motion_n numbering must continue incrementally with each new function, regardless of what motion is loaded or branched.

        Follow these coding rules strictly:
        - Include inline comments explaining how each API call relates to the user instruction.
        - Clearly explain how your code preserves the intent of the instruction.

        Generate Python function for each new instruction, using only the defined Motion API. 
        You don't need to give me all previous written codes. 
        Just give me codes for given current user input instruction.
        Note that current sequence has 150 frames. (But can be changed by insturction)
        
        You are starting from a fresh state starting from "motion_1". 
        Remember initialize a new Motion_DB() only once just before 1st edit. 
        
        Don't forget, you still have to increment the motion number, even after 'clear all' related instruction.
        
        Now, let's get to it. Ready? 
    ''')

    prompt_sequence.append("Yes.")

    return prompt_sequence













