import torch
import os
import pickle
import numpy as np
from datetime import datetime
import sys
from argparse import ArgumentParser
import argparse
from src.FaceMEO.llm.motion_io import Motion_DB
from src.FaceMEO.llm.motion import FacialMotion
sys.path.append('./livelink_MEAD/') # added to access face_model_io in data_generation.py file
from data.livelink_MEAD import face_model_io
from src.FaceMEO.openai_wrapper import read_progprompt_0, read_progprompt, get_incontext, query_model
from src.disentanglement.train import Runner
from src.disentanglement.test import direct_decoding, direct_decoding_only_exp
from parser_util import IEFA_args

sys.path.append('/source/inyup/IEFA/data/livelink_MEAD')
from util import v_render_sequence_meshes, mux_audio_video, concat_videos, np_apply_gaussian_filter, new_np_gaussian_kernel1d

from data.livelink_MEAD.util import bshp_2_vtx

db = Motion_DB()

def run_pipeline(prompt_sequence, context, autorun=False, user_input : str = None):
    
    if not autorun:
        user_input = input("\nYou: ")
    
    prompt = user_input

    prompt_sequence.append("# " + prompt + "\n")
    error_prompt_sequence = prompt_sequence

    c, r, context = query_model(prompt_sequence, error_prompt_sequence, 0, context)
    # import pdb;pdb.set_trace()
    prompt_sequence.append(c)
    print(f"Chatbot: {c} \n")
    
    return context
    
def run_pipeline_STARLAB_data_construction(base_prompt_sequence, context, autorun=False, user_input : str = None):
    if not autorun:
        user_input = input("\nYou: ")
    
    prompt = user_input.strip()

    prompt_sequence = base_prompt_sequence.copy()
    
    prompt_sequence.append("# " + prompt + "\n")
    # 에러 프롬프트도 이번 instruction 안에서만 누적
    error_prompt_sequence = prompt_sequence.copy()
    
    c, r, context = query_model(prompt_sequence, error_prompt_sequence, 0, context)
    
    # prompt_sequence.append(c)
    print(f"Chatbot: {c} \n")
    return context


if __name__ == "__main__":
    
    
    
    print("---------------- Running 1st year STARLAB data generation pipeline ------------------")
    
    ## args needed
    prompt_sequence = []
    context = {}
    face_model = face_model_io.load_face_model('/source/inyup/ICT-FaceKit/FaceXModel') # switched to util_fast 
    final_mux_result_path = ""
    #################
    render = True # switch on and off
    render_only_result = False
    ####################
    motion_id = 1
    run_pipeline_flag = False # switch on and off
    save_vtx = True # switch on and off
    save_bshp = False # switch on and off
    
    if run_pipeline_flag:
        hparams = IEFA_args()
        device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        init_prompt = "Always Initialize a new motion with total length of frame 100. And save it as motion_1. "
        inst_pair_path = "/source/inyup/IEFA/data/STARLAB/2nd-year/inst-pair-100-run-v2.txt"
        # inst_pair_path = "/source/inyup/IEFA/data/FaceBlendshapeGen/expression-video-instruction.txt"
        with open(inst_pair_path, 'r') as f:
            inst_pairs = f.readlines()
            inst_pairs = [line.strip() for line in inst_pairs]
        base_prompt_sequence = []
        base_prompt_sequence = get_incontext(base_prompt_sequence)
    else:
        bshp_anim_source_path = "/source/inyup/IEFA/data/STARLAB/2nd-year/anim-bshp-param"
        from glob import glob
        inst_pair_paths = sorted(glob(os.path.join(bshp_anim_source_path, "*_exp_v2_*.npy")))
        # import pdb;pdb.set_trace()
    
    for id in range(8):
        # if id < 7:
        #     print(f"skipping ids {id}...")
        #     continue
        
        ## deform id
        id_param_path = f"/source/inyup/IEFA/id_{id}_param.npy"
        if os.path.exists(id_param_path):
            id_param = np.load(id_param_path)
        else:
            id_param = np.random.randn(100)
            np.save(f"./id_{id}_param.npy", id_param)
        
        ######################
        ## w/ running pipeline
        if run_pipeline_flag:
            total_frames = 100   
            for idx, inst_pair in enumerate(inst_pairs):
                
                
                    # if id == 7 and idx+1 < 41:
                    #     print(f"skipping instrunctions {idx}...")
                    #     continue
                    print(f"----- Running for {idx+1}th instruction: {inst_pair}")
                    inst_pair_tweaked = init_prompt + inst_pair

                    context = run_pipeline_STARLAB_data_construction(base_prompt_sequence, context, autorun=True, user_input=inst_pair_tweaked) # 1. queries GPT / 2. run GPT created code
                    
                    motion_key = f"motion_{motion_id}"
                    motion_info = context.get('db').load_motion(motion_key, return_dict=True) # to keep track of exec() execution's state of Motion_DB()
                    motion = FacialMotion(motion_info, total_frames=total_frames) # specify total_frames here if needed
                    
                    exp_bshp_anim_seq = motion.output_animation_seq
                    # filename = str(idx+1).zfill(3) + f"_IEFA_FaceBlendshapeGen_v1_{id+1}"
                    filename = str(idx+1).zfill(3) + f"_IEFA_STARLAB_exp_v2_{id+1}"
                    # np.save(f"/source/inyup/IEFA/data/test/result/vtx/{filename}_bshp_anim.npy", exp_bshp_anim_seq)
                    
                    exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq, face_model, id_param=id_param)
                    bshp_tmp_video_path = direct_decoding_only_exp(hparams=hparams, 
                                                                motion_num=idx, 
                                                                exp_vtx_anim_seq=exp_vtx_anim_seq, 
                                                                bshp_tmp_video_name=f"{filename}_bshp_anim_vid", 
                                                                face_model=face_model)
        #######################
        ## w/o running pipeline
        else:
            for idx, inst_pair_path in enumerate(inst_pair_paths):
                print(f"----- Direct Running for {idx+1}th instruction -----")
                filename = str(idx+1).zfill(3) + f"_IEFA_STARLAB_exp_v2_{id+1}"
                if save_bshp:
                    np.save(f"/source/inyup/IEFA/data/test/result/vtx/{filename}_bshp_anim.npy", exp_bshp_anim_seq)
                exp_bshp_anim_seq = np.load(inst_pair_path)
                exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq, face_model, id_param=id_param)
                bshp_tmp_video_path = direct_decoding_only_exp( 
                                                            exp_vtx_anim_seq=exp_vtx_anim_seq, 
                                                            bshp_tmp_video_name=f"{filename}_bshp_anim_vid", 
                                                            face_model=face_model,
                                                            save_vtx=save_vtx,
                                                            save_vtx_dir="/source/inyup/IEFA/data/STARLAB/2nd-year/anim-id-vtx-v2",
                                                            save_video_dir="/source/inyup/IEFA/data/STARLAB/2nd-year/anim-id-vid-v2")