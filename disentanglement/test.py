"""
###########
### log ###
###########
## 2024-08-21
modified version of test.py of original FaceCLIP 
: To render ICT-FaceKit mesh
## 2024-09-02
added direct_decoding() for use in run_IEFA.py
## 2025-03-11
added direct_decoding_o3d() for faster render
"""

import sys
sys.path.insert(0, '../')
import os
import torch
import pickle
import numpy as np
from datetime import datetime
sys.path.append('../../')
from parser_util import disentangle_args
from disentanglement.train_faceclip import Runner
from render import render_from_vertex
sys.path.append('/input/inyup/IEFA/data/livelink_MEAD')
from util import v_render_sequence_meshes, mux_audio_video, concat_videos, np_apply_gaussian_filter, new_np_gaussian_kernel1d
# from util_fast import v_render_sequence_meshes, mux_audio_video, concat_videos
## TODO 
## - (DONE) regard this code and parser_util.py to configure 'add_IEFA_options' function

def direct_decoding_only_exp( 
                    exp_vtx_anim_seq,
                    bshp_tmp_video_name, 
                    face_model = None,
                    save_vtx = False,
                    save_vtx_dir = './',
                    save_video_dir = './'):
    ## render naive expression parameter
    # bshp_tmp_video_name = 'bshp_tmp'
    if save_vtx:
        vtx_path = bshp_tmp_video_name.replace('_vid', '_vtx') + '.npy' 
        # import pdb;pdb.set_trace()
        vtx_full_path = os.path.join(save_vtx_dir, vtx_path)
        np.save(vtx_full_path, exp_vtx_anim_seq)
    v_render_sequence_meshes(Vs_path = exp_vtx_anim_seq,
                                video_name = bshp_tmp_video_name,
                                mode='shade',
                                out_root_dir=save_video_dir, 
                                face_model=face_model)
        
def direct_decoding(hparams, 
                    motion_num, 
                    con_vtx_anim_seq, 
                    exp_vtx_anim_seq, 
                    src_vid_path, 
                    face_model = None, 
                    audio_path = None, 
                    runner= None, 
                    render = True, 
                    render_only_result=False, 
                    sentence_number=None, 
                    edit_number=None):
    with torch.no_grad():
        device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        # import pdb;pdb.set_trace()
        if sentence_number != None:
            filename = f"s{str(12 - sentence_number)}_e{edit_number+1}"
        else:
            filename = datetime.now().date().strftime("%Y%m%d") + "_" + str(motion_num)
        
        os.makedirs(f'{hparams.root_dir + hparams.save_vtx_dir}', exist_ok=True)
        os.makedirs(f'{hparams.root_dir + hparams.save_video_dir}', exist_ok=True)
        
        vtx_path = f'{hparams.root_dir + hparams.save_vtx_dir}/{filename}_dis_{hparams.model_num}.npy'
        vid_path = f'{hparams.root_dir + hparams.save_video_dir}/{filename}_dis_{hparams.model_num}.mp4'
        
        vtx_con = torch.from_numpy(con_vtx_anim_seq).to(device).float()
        vtx_exp = torch.from_numpy(exp_vtx_anim_seq).to(device).float()
        pred_vtx = runner.autoencoder.reconstruct([vtx_con, vtx_exp])

        pred_vtx = pred_vtx.squeeze()
        pred = pred_vtx.cpu().numpy()
        
        ########################
        ## apply gaussian filter
        ########################
        # pred = np_apply_gaussian_filter(pred)
        # pred = new_np_gaussian_kernel1d(pred) * 0.7 # to render into a bit smaller sized model
        # pred = new_np_gaussian_kernel1d(pred) 

        if sentence_number != None:
            vtx_path = vtx_path.replace(f'{filename}_dis_{hparams.model_num}', filename)
            np.save(vtx_path, pred) 
        else:
            np.save(vtx_path, pred) 
        
        ##############
        ## save video
        ##############
        if sentence_number != None:
            src_tmp_video_name = f'tmp_s{12-sentence_number}'
        else: 
            src_tmp_video_name = 'tmp'
        src_tmp_video_path = f'{src_tmp_video_name}.mp4'

        if render:
            if not render_only_result:
                ## render source animation 
                # if src_vid_path == "": # if it's neutral speech animation,
                if not os.path.exists(os.path.join(hparams.root_dir + hparams.save_video_dir, src_tmp_video_path)):
                    src_tmp_video_path = os.path.join(hparams.root_dir + hparams.save_video_dir, src_tmp_video_path)
                    v_render_sequence_meshes(Vs_path = con_vtx_anim_seq,
                                            video_name=src_tmp_video_name,
                                            mode='shade',
                                            out_root_dir=hparams.root_dir + hparams.save_video_dir, 
                                            face_model=face_model)
                else: # if it's not, use as it is
                #     src_tmp_video_path = src_vid_path
                    src_tmp_video_path = os.path.join(hparams.root_dir + hparams.save_video_dir, src_tmp_video_path)
                
                ## render naive expression parameter
                bshp_tmp_video_name = 'bshp_tmp'
                v_render_sequence_meshes(Vs_path = exp_vtx_anim_seq,
                                            video_name=bshp_tmp_video_name,
                                            mode='shade',
                                            out_root_dir=hparams.root_dir + hparams.save_video_dir, 
                                            face_model=face_model)
                bshp_tmp_video_path = os.path.join(hparams.root_dir + hparams.save_video_dir, f'{bshp_tmp_video_name}.mp4')
                
                ## render output animation
                video_name = os.path.basename(vid_path).split('.')[0]
                final_mux_result_path = vid_path.replace('.mp4','_mux.mp4')
                v_render_sequence_meshes(Vs_path = vtx_path,
                                        video_name=video_name,
                                        mode='shade',
                                        out_root_dir=hparams.root_dir + hparams.save_video_dir, 
                                        face_model=face_model)
                if audio_path != None:
                    ## mux video
                    mux_audio_video(audio_path, vid_path, final_mux_result_path)
                    ## concat videos 
                    concat_videos([src_tmp_video_path, bshp_tmp_video_path, final_mux_result_path], final_mux_result_path.replace('.mp4', 'cat.mp4'))
                    ## remove materials
                    # if src_vid_path == "":
                    #     os.remove(src_tmp_video_path)
                    os.remove(vid_path)
                    os.remove(bshp_tmp_video_path)
                    os.remove(final_mux_result_path) # delete final mux result as well 
                    print("[{}] muxcat video vtx file saved. Frame length: {}".format(filename, pred.shape[0]))
                else:
                    
                    ## for unseen audio only 
                    concat_videos([src_tmp_video_path, bshp_tmp_video_path, vid_path], vid_path.replace('.mp4', 'cat.mp4'))

                    print("[{}] video and vtx file saved. Frame length: {}".format(filename, pred.shape[0]))

            else:
                ## render output animation
                video_name = os.path.basename(vid_path).split('.')[0]
                final_mux_result_path = vid_path.replace('.mp4','_mux.mp4')
                v_render_sequence_meshes(Vs_path = vtx_path,
                                        video_name=video_name,
                                        mode='shade',
                                        out_root_dir=hparams.root_dir + hparams.save_video_dir, 
                                        face_model=face_model)
                if audio_path != None:
                    ## mux video
                    mux_audio_video(audio_path, vid_path, final_mux_result_path)
                    os.remove(vid_path)
                    print("[{}] mux single output video vtx file saved. Frame length: {}".format(filename, pred.shape[0]))
                else:
                    print("[{}] video and vtx file saved. Frame length: {}".format(filename, pred.shape[0]))
         
         
        ###################################       
        if render:
            return final_mux_result_path, pred # outputs numpy array   
        else:
            return pred

def test(hparams, runner, vtx_dict, audio_path):
    with torch.no_grad():
        device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        # filename = os.path.splitext(hparams.test_audio)[0]
        filename = vtx_dict['filename']
        
        os.makedirs(f'{hparams.root_dir + hparams.save_vtx_dir}', exist_ok=True)
        os.makedirs(f'{hparams.root_dir + hparams.save_video_dir}', exist_ok=True)
        
        vtx_path = f'{hparams.root_dir + hparams.save_vtx_dir}/{filename}_dis{hparams.model_num}.npy'
        vid_path = f'{hparams.root_dir + hparams.save_video_dir}/{filename}_dis{hparams.model_num}.mp4'
        # import pdb;pdb.set_trace()
        vtx_con = torch.from_numpy(vtx_dict['con']).to(device).float()
        vtx_exp = torch.from_numpy(vtx_dict['exp']).to(device).float()
        pred_vtx = runner.autoencoder.reconstruct([vtx_con, vtx_exp])

        pred_vtx = pred_vtx.squeeze()
        pred = pred_vtx.cpu().numpy()

        # save video
        np.save(vtx_path, pred)
        # render_from_vertex(hparams, vtx_path, audio_path, vid_path)
        video_name = os.path.basename(vid_path).split('.')[0]
        final_mux_result_path = vid_path.replace('.mp4','_mux.mp4')
        v_render_sequence_meshes(Vs_path = vtx_path,video_name=video_name,mode='shade', out_root_dir=hparams.root_dir + hparams.save_video_dir)
        mux_audio_video(audio_path, vid_path, final_mux_result_path)
        os.remove(vid_path)
        print("[{}] Saved. Frame length: {}".format(filename, pred.shape[0]))


if __name__ == '__main__':
    hparams = disentangle_args()
    device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")

    print(" ------ Test Model : {} ------- ".format(hparams.model_num))
    model_path = f"{os.path.join(hparams.root_dir , hparams.save_dir)}/{hparams.model_num}.pth"
    runner = Runner(hparams)
    # import pdb;pdb.set_trace()
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()

    # Test file
    f = open(hparams.root_dir + hparams.vtx_dtw_path, 'rb')
    vtx = pickle.load(f)
    f.close()

    ##############################
    ## different con different exp
    content, emotion = 'angry', 'happy'
    # content, emotion = 'angry', 'surprised'
    # content, emotion = 'angry', 'contempt'      
    # content, emotion = 'angry', 'fear'
    # content, emotion = 'happy', 'angry'
    # content, emotion = 'happy', 'disgusted' # for '12' disgusted is 030 whereas others are 003
    # content, emotion = 'angry', 'angry'
    # content, emotion = 'angry', 'disgusted'

    # content, emotion = 'neutral', 'disgusted'
    vtx_dict = {}
    # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    ## cus_eus
    # con_list = [content, '12', '003']
    # exp_list = [emotion, '12', '003']
    # ## cus_es
    # con_list = [content, '12', '003']
    # exp_list = [emotion, '11', '030']
    ## cs_es
    # con_list = [content, '11', '030']
    # exp_list = [emotion, '11', '030']
    ## cs_eus
    con_list = [content, '12', '003']
    exp_list = [emotion, '12', '003']
    
    filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    vtx_dict['con'] = vtx[f'{con_list[1]}'][f'{content}_3_M003_front_{content}_3_{con_list[-1]}']
    # vtx_dict['con'] = vtx[f'{con_list[1]}'][f'{content}_1_M003_front_{content}_1_{con_list[-1]}'] # if neutral
    vtx_dict['exp'] = vtx[f'{exp_list[1]}'][f'{emotion}_3_M003_front_{emotion}_3_{exp_list[-1]}']
    
    ## if any seqence longer than other, just clip it
    if vtx_dict['con'].shape[0] < vtx_dict['exp'].shape[0]:
        vtx_dict['exp'] = vtx_dict['exp'][:vtx_dict['con'].shape[0]]
    else:
        vtx_dict['con'] = vtx_dict['con'][:vtx_dict['exp'].shape[0]]
        
    vtx_dict['filename'] = '{}'.format(filename)
    test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    test(hparams, runner, vtx_dict, test_filepath)
    
    # ##########################
    # ## same con different exp
    # content, emotion = 'angry', 'happy'
    # vtx_dict = {}
    # # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    # con_list = [content, '3', '003']
    # exp_list = [emotion, '3', '021']
    # filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'    # import pdb;pdb.set_trace()
    # # import pdb;pdb.set_trace()
    
    # vtx_dict['con'] = vtx['4'][f'{content}_{con_list[1]}_M003_front_{content}_3_{con_list[-1]}']
    # vtx_dict['exp'] = vtx['5'][f'{emotion}_{exp_list[1]}_M003_front_{emotion}_3_{exp_list[-1]}'][:vtx_dict['con'].shape[0]]
    # vtx_dict['filename'] = '{}'.format(filename)
    # test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    # test(hparams, runner, vtx_dict, test_filepath)
    
    #################################
    ## same con same exp (self recon)
    # content, emotion = 'angry', 'angry'
    # vtx_dict = {}
    # # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    # con_list = [content, '3', '003']
    # exp_list = [emotion, '3', '003']
    # filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'
    # # import pdb;pdb.set_trace()
    
    # vtx_dict['con'] = vtx['4'][f'{content}_{con_list[1]}_M003_front_{content}_3_{con_list[-1]}']
    # vtx_dict['exp'] = vtx['4'][f'{emotion}_{exp_list[1]}_M003_front_{emotion}_3_{exp_list[-1]}'][:vtx_dict['con'].shape[0]]
    # vtx_dict['filename'] = '{}'.format(filename)
    # test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    # test(hparams, runner, vtx_dict, test_filepath)
