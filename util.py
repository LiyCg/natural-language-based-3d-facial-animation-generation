import sys
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial

from glob import glob
from tqdm import tqdm

import openmesh as om 
sys.path.append('./livelink_MEAD/') # added to access face_model_io in data_generation.py file
import face_model_io

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import moviepy.editor as mp
from IPython.display import Video

sys.path.append('./../')
sys.path.append('/source/inyup/IEFA/data/')

# Define the mapping between morph target indices and their corresponding names
morph_targets = [
    "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R", 
    "browOuterUp_L", "browOuterUp_R", "cheekPuff_L", "cheekPuff_R", 
    "cheekSquint_L", "cheekSquint_R", "eyeBlink_L", "eyeBlink_R", 
    "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R", 
    "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", 
    "eyeSquint_L", "eyeSquint_R", "eyeWide_L", "eyeWide_R", 
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", 
    "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R", 
    "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", 
    "mouthPress_L", "mouthPress_R", "mouthPucker", "mouthRight", 
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", 
    "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R", 
    "mouthUpperUp_L", "mouthUpperUp_R", "noseSneer_L", "noseSneer_R"
]

# Define the mapping from CSV columns to morph target names
csv_to_morph = {
    "EyeBlinkLeft": "eyeBlink_L", "EyeBlinkRight": "eyeBlink_R",
    "EyeLookDownLeft": "eyeLookDown_L", "EyeLookDownRight": "eyeLookDown_R",
    "EyeLookInLeft": "eyeLookIn_L", "EyeLookInRight": "eyeLookIn_R",
    "EyeLookOutLeft": "eyeLookOut_L", "EyeLookOutRight": "eyeLookOut_R",
    "EyeLookUpLeft": "eyeLookUp_L", "EyeLookUpRight": "eyeLookUp_R",
    "EyeSquintLeft": "eyeSquint_L", "EyeSquintRight": "eyeSquint_R",
    "EyeWideLeft": "eyeWide_L", "EyeWideRight": "eyeWide_R",
    "JawForward": "jawForward", "JawRight": "jawRight", "JawLeft": "jawLeft",
    "JawOpen": "jawOpen", "MouthClose": "mouthClose", "MouthFunnel": "mouthFunnel",
    "MouthPucker": "mouthPucker", "MouthRight": "mouthRight", "MouthLeft": "mouthLeft",
    "MouthSmileLeft": "mouthSmile_L", "MouthSmileRight": "mouthSmile_R",
    "MouthFrownLeft": "mouthFrown_L", "MouthFrownRight": "mouthFrown_R",
    "MouthDimpleLeft": "mouthDimple_L", "MouthDimpleRight": "mouthDimple_R",
    "MouthStretchLeft": "mouthStretch_L", "MouthStretchRight": "mouthStretch_R",
    "MouthRollLower": "mouthRollLower", "MouthRollUpper": "mouthRollUpper",
    "MouthShrugLower": "mouthShrugLower", "MouthShrugUpper": "mouthShrugUpper",
    "MouthPressLeft": "mouthPress_L", "MouthPressRight": "mouthPress_R",
    "MouthLowerDownLeft": "mouthLowerDown_L", "MouthLowerDownRight": "mouthLowerDown_R",
    "MouthUpperUpLeft": "mouthUpperUp_L", "MouthUpperUpRight": "mouthUpperUp_R",
    "BrowDownLeft": "browDown_L", "BrowDownRight": "browDown_R",
    "BrowInnerUp": "browInnerUp_L",  # Assuming both brows move together for inner up
    "BrowOuterUpLeft": "browOuterUp_L", "BrowOuterUpRight": "browOuterUp_R",
    "CheekPuff": "cheekPuff_L", "CheekPuff": "cheekPuff_R",  # Assuming both cheeks puff together
    "CheekSquintLeft": "cheekSquint_L", "CheekSquintRight": "cheekSquint_R",
    "NoseSneerLeft": "noseSneer_L", "NoseSneerRight": "noseSneer_R"
}

def calibrated_extract_expression_parameter(raw_capture_csv_path = "", cal_capture_csv_path = "", neut_capture_csv_path = "", output_path = ""):

    # Read the CSV file
    raw_df = pd.read_csv(raw_capture_csv_path)
    cal_df = pd.read_csv(cal_capture_csv_path)
    neutral_df = pd.read_csv(neut_capture_csv_path)
    if len(neutral_df) == 1:
        # If neutral_data has only one row, broadcast it across all rows
        neutral_df = pd.concat([neutral_df] * len(raw_df), ignore_index=True)
    elif neutral_df.shape != raw_df.shape:
        raise ValueError("Neutral data dimensions do not match the Raw or Cal data!")
        
    raw_numeric = raw_df.iloc[:, 2:]  # Select all columns starting from the 3rd
    cal_numeric = cal_df.iloc[:, 2:]
    neutral_numeric = neutral_df.iloc[:, 2:]
    
    correct_numeric = raw_numeric - cal_numeric
    correct_numeric = correct_numeric + neutral_numeric 
    correct_df = pd.concat([raw_df[['Timecode', 'BlendshapeCount']], correct_numeric], axis=1)

    # Initialize a list to store the expression parameter vectors
    expression_vectors = [] # animation sequences

    # Iterate over each row in the dataframe
    for index, row in correct_df.iterrows():
        expression_vector = np.zeros((len(morph_targets))) # 53 morph targets
        for csv_col, morph_name in csv_to_morph.items():
            if morph_name in morph_targets:
                target_index = morph_targets.index(morph_name)
                expression_vector[target_index] = row[csv_col]
        expression_vectors.append(expression_vector)

    expression_vectors = np.array(expression_vectors) # convert to np araay
    
    print(f"Extracted from {os.path.basename(raw_capture_csv_path)} and {os.path.basename(cal_capture_csv_path)} has shape of {expression_vectors.shape}")
    # print(expression_vectors)
    np.save(output_path, expression_vectors)
    
    return expression_vectors


def extract_expression_parameter(livelink_capture_csv_path = "", output_path = ""):

    # Read the CSV file
    df = pd.read_csv(livelink_capture_csv_path)

    # import pdb;pdb.set_trace()
    # Extract the timecode and blendshape columns (assuming timecode is the first column)
    timecode_column = df.columns[0]  
    blendshape_columns = df.columns[2:]  # Adjust based on actual CSV structure, 61 number of columns

    # Initialize a list to store the expression parameter vectors
    expression_vectors = [] # animation sequences

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        expression_vector = np.zeros((len(morph_targets))) # 53 morph targets
        for csv_col, morph_name in csv_to_morph.items():
            if morph_name in morph_targets:
                target_index = morph_targets.index(morph_name)
                expression_vector[target_index] = row[csv_col]
        expression_vectors.append(expression_vector)

    expression_vectors = np.array(expression_vectors) # convert to np araay
    
    print(f"Extracted from {os.path.basename(livelink_capture_csv_path)} has shape of {expression_vectors.shape}")
    # print(expression_vectors)
    np.save(output_path, expression_vectors)
    
    return expression_vectors

"""
version just gives vertices
"""
def v_render_sequence_meshes(Vs_path : np.array = None, 
                           video_name="",
                           bg_black=False,
                           fps=30,
                           face_only=True,
                           remove_axis=True,
                           show_angles=False,
                           figsize=(6,6),
                           mode="shade",
                           light_dir=np.array([0,0,1]),
                           linewidth = 1,
                           out_root_dir = "./result",
                           face_model = None,
                           ):
    
    # expression_parameters = np.load(expression_parameters_path)
    if face_model == None:
        face_model = face_model_io.load_face_model('/input/inyup/ICT-FaceKit/FaceXModel')
    
    frame_vertices = [] # [N, 53]
    if type(Vs_path) == str: # if path
        Vs = np.load(Vs_path)
    else:
        Vs = Vs_path
    frame_len = Vs.shape[0]
    Vs = Vs.reshape(frame_len,-1,3)
    # import pdb;pdb.set_trace()
    for frame_idx in range(frame_len):
        # # blendshape 53
        parameter = np.zeros((53))
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        # import pdb;pdb.set_trace()
        
        deformed_vertices = face_model._deformed_vertices.copy()
        deformed_vertices[:9409,] = Vs[frame_idx][:,:]
        frame_vertices.append(deformed_vertices)

    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if show_angles:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax2 = fig.add_subplot(132, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax3 = fig.add_subplot(133, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        if remove_axis:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
    else:
        fig = plt.figure(figsize=figsize)
        _r = figsize[0] / figsize[1]
        fig_xlim = [-_r,_r]
        fig_ylim = [-1,+1]
        single_ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
        if remove_axis:
            single_ax.axis('off')

    if face_only:
        ## face only
        v_idx, f_idx = 9409, 9230
    else:
        ## face + head + neck
        v_idx, f_idx = 11248, 11144

    # This is quad mesh!
    quad_F  = face_model._generic_neutral_mesh.face_vertex_indices()[:f_idx] # [N,4]
    tri_faces_1 = quad_F[:, [0, 1, 2]] # [N,3]
    tri_faces_2 = quad_F[:, [0, 2, 3]] # [N,3]
    F = np.vstack([tri_faces_1, tri_faces_2]) # [2N,3]
    Vs = frame_vertices # [V,3]

    # Pre-calculated transformation matrices for three views
    if mode == "mesh":
        proj = perspective(25, 1, 1, 100) # for mesh
        model = translate(0, 0, -2.5) 
    else:
        proj = ortho(-12,12,-12,12,1,100) # for shade
        model = translate(0, 0, -2.5) 
        
    if show_angles: # if wanna render with different angles, this should be coupled with 'mesh' mode
        MVPs = [
            proj @ model,
            proj @ model @ yrotate(-30),
            proj @ model @ yrotate(-90)
        ]
    else: # if only front
        MVP = proj @ model,
        MVP = MVP[0] # IDK, but it is tupled
        # import pdb;pdb.set_trace() 
    
    def render_mesh(ax, V, MVP, F):
        # quad to triangle
        # import pdb;pdb.set_trace()
        if mode == "mesh":
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=True)
        else:
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=False)
        
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode == "shade":
            C = calc_face_norm(V, F) @ model[:3,:3].T # [3,3] -----> contains the transformed face normals
            I = np.argsort(Z) # -----------------------------------> depth sorting 
            T, C = T[I, :], C[I, :] # ensures that triangles are rendered from back to front
            
            NI = np.argwhere(C[:, 2] > 0).squeeze() # -------------> culling w/ normal,  checks the z-component of the normals, if  positive, the face is facing towards the camera.
            T, C = T[NI, :], C[NI, :] # only extracts faces facing front 
            
            C = np.clip((C @ light_dir), 0, 1) # ------------------> cliping range 0 - 1, resulting in light intensity value
            C = C[:, np.newaxis].repeat(3, axis=-1) # making RGB channels
            C = np.clip(C, 0, 1) # intensity values remain within the range [0, 1]
            C = C*0.6+0.15 # reduces the intensity by 40%, ensuring brightest are not too bright / adds a small base value to the intensity, ensuring that  darkest have some brightness
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C  = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        
        ax.add_collection(collection)

    def update(V):
        # Cleanup previous collections
        if show_angles:
            for _ax in [ax1, ax2, ax3]:
                for coll in _ax.collections:
                    coll.remove()
            
            # Render meshes for all views
            for _ax, mvp in zip([ax1, ax2, ax3], MVPs):
                render_mesh(_ax, V, mvp, F)
                
            return ax1.collections + ax2.collections + ax3.collections
        
        else:
            for coll in single_ax.collections:
                coll.remove()
            # import pdb;pdb.set_trace()
            render_mesh(single_ax, V, MVP, F)
            
            return single_ax.collections

    ani = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering frames", ncols=100), blit=True) # Vs[56:202]

    ### can be saved in difference format
    os.makedirs(out_root_dir, exist_ok=True)
    ani.save(f'{out_root_dir}/{video_name}.mp4', writer='ffmpeg', fps=fps)
    plt.close()


def render_sequence_meshes(expression_parameters_path= "", 
                           video_name="",
                           bg_black=False,
                           fps=30,
                           face_only=True,
                           remove_axis=True,
                           show_angles=True,
                           figsize=(6,6),
                           mode="mesh",
                           light_dir=np.array([0,0,1]),
                           linewidth = 1,
                           out_root_dir = "./result",
                           face_model = None,
                           ):
    
    expression_parameters = np.load(expression_parameters_path)
    if face_model == None:
        face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    frame_vertices = [] # [N, 53]
    
    for frame_idx, parameter in enumerate(expression_parameters):
        # # blendshape 53
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        # # Write the deformed mesh
        # face_model_io.write_deformed_mesh('/data/sihun/arkit_CSH/sample_identity_arkit_frame_{:06d}.obj'.format(idx), face_model)
        # # om.write_mesh(file_path, face_model._deformed_mesh, halfedge_tex_coord = True)
        frame_vertices.append(face_model._deformed_vertices.copy())


    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if show_angles:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax2 = fig.add_subplot(132, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax3 = fig.add_subplot(133, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        if remove_axis:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
    else:
        fig = plt.figure(figsize=figsize)
        _r = figsize[0] / figsize[1]
        fig_xlim = [-_r,_r]
        fig_ylim = [-1,+1]
        single_ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
        if remove_axis:
            single_ax.axis('off')

    if face_only:
        ## face only
        v_idx, f_idx = 9409, 9230
    else:
        ## face + head + neck
        v_idx, f_idx = 11248, 11144

    # This is quad mesh!
    quad_F  = face_model._generic_neutral_mesh.face_vertex_indices()[:f_idx] # [N,4]
    tri_faces_1 = quad_F[:, [0, 1, 2]] # [N,3]
    tri_faces_2 = quad_F[:, [0, 2, 3]] # [N,3]
    F = np.vstack([tri_faces_1, tri_faces_2]) # [2N,3]
    Vs = frame_vertices # [V,3]

    # Pre-calculated transformation matrices for three views
    if mode == "mesh":
        proj = perspective(25, 1, 1, 100) # for mesh
        model = translate(0, 0, -2.5) 
    else:
        proj = ortho(-12,12,-12,12,1,100) # for shade
        model = translate(0, 0, -2.5) 
        
    if show_angles: # if wanna render with different angles, this should be coupled with 'mesh' mode
        MVPs = [
            proj @ model,
            proj @ model @ yrotate(-30),
            proj @ model @ yrotate(-90)
        ]
    else: # if only front
        MVP = proj @ model,
        MVP = MVP[0] # IDK, but it is tupled
        # import pdb;pdb.set_trace() 
    
    def render_mesh(ax, V, MVP, F):
        # quad to triangle
        # import pdb;pdb.set_trace()
        if mode == "mesh":
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=True)
        else:
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=False)
        
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode == "shade":
            C = calc_face_norm(V, F) @ model[:3,:3].T # [3,3] -----> contains the transformed face normals
            I = np.argsort(Z) # -----------------------------------> depth sorting 
            T, C = T[I, :], C[I, :] # ensures that triangles are rendered from back to front
            
            NI = np.argwhere(C[:, 2] > 0).squeeze() # -------------> culling w/ normal,  checks the z-component of the normals, if  positive, the face is facing towards the camera.
            T, C = T[NI, :], C[NI, :] # only extracts faces facing front 
            
            C = np.clip((C @ light_dir), 0, 1) # ------------------> cliping range 0 - 1, resulting in light intensity value
            C = C[:, np.newaxis].repeat(3, axis=-1) # making RGB channels
            C = np.clip(C, 0, 1) # intensity values remain within the range [0, 1]
            C = C*0.6+0.15 # reduces the intensity by 40%, ensuring brightest are not too bright / adds a small base value to the intensity, ensuring that  darkest have some brightness
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C  = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        
        ax.add_collection(collection)

    def update(V):
        # Cleanup previous collections
        if show_angles:
            for _ax in [ax1, ax2, ax3]:
                for coll in _ax.collections:
                    coll.remove()
            
            # Render meshes for all views
            for _ax, mvp in zip([ax1, ax2, ax3], MVPs):
                render_mesh(_ax, V, mvp, F)
                
            return ax1.collections + ax2.collections + ax3.collections
        
        else:
            for coll in single_ax.collections:
                coll.remove()
            # import pdb;pdb.set_trace()
            render_mesh(single_ax, V, MVP, F)
            
            return single_ax.collections

    ani = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering frames", ncols=100), blit=True) # Vs[56:202]

    ### can be saved in difference format
    os.makedirs(out_root_dir, exist_ok=True)
    ani.save(f'{out_root_dir}/{video_name}.mp4', writer='ffmpeg', fps=fps)
    plt.close()
    
def trim_ict_parameter(reference_video_path, expression_parameters_path, output_path, smoothing_sigmna = 1):
    
    ## to follow referene video frame number
    video = mp.VideoFileClip(reference_video_path)
    ref_num_frames = int(video.fps * video.duration)
    
    # import pdb;pdb.set_trace()
    expression_parameters = np.load(expression_parameters_path)
    
    ## apply gaussian smoothing
    # from scipy.ndimage import gaussian_filter1d
    # smoothed_parameters = gaussian_filter1d(expression_parameters, sigma=smoothing_sigmna, axis=0)
    
    ## compute delta vectors 
    tmp_expression_parameters = (expression_parameters * 1000).astype(int)
    delta_vectors = np.abs(np.diff(tmp_expression_parameters, axis=0))    
    ## sum of delta vectors 
    delta_sums = np.sum(delta_vectors, axis=1)
    second_delta_sums = np.abs(np.diff(delta_sums, axis=0))
    # third_delta_sums = np.abs(np.diff(second_delta_sums, axis=0))
    
    # change_points = np.where(third_delta_sums > np.mean(third_delta_sums))[0]
    # change_points = np.argsort(second_delta_sums)[-2:][::-1]
    change_point = second_delta_sums.argmax()
    
    # if len(change_points) == 0:
    #     print("no significant change points detected")
    #     return

    ## find the first significant change point
    end_idx = change_point + 1 
    start_idx = end_idx - ref_num_frames 
    
    new_expression_parameters = expression_parameters[start_idx : end_idx]    
    
    np.save(output_path, new_expression_parameters)
    print(f"strat index : {start_idx} / shape of trimmed expression parameter : f{new_expression_parameters.shape}")
    
"""
version just gives numpy array
"""  
def _trim_ict_parameter(reference_video_path, expression_parameters, output_path, smoothing_sigmna = 1):
    
    ## to follow referene video frame number
    video = mp.VideoFileClip(reference_video_path)
    ref_num_frames = int(video.fps * video.duration)
    
    ## apply gaussian smoothing
    # from scipy.ndimage import gaussian_filter1d
    # smoothed_parameters = gaussian_filter1d(expression_parameters, sigma=smoothing_sigmna, axis=0)
    
    ## compute delta vectors 
    tmp_expression_parameters = (expression_parameters * 1000).astype(int)
    delta_vectors = np.abs(np.diff(tmp_expression_parameters, axis=0))    
    ## sum of delta vectors 
    delta_sums = np.sum(delta_vectors, axis=1)
    second_delta_sums = np.abs(np.diff(delta_sums, axis=0))
    # third_delta_sums = np.abs(np.diff(second_delta_sums, axis=0))
    
    # change_points = np.where(third_delta_sums > np.mean(third_delta_sums))[0]
    # change_points = np.argsort(second_delta_sums)[-10:][::-1]
    change_point = second_delta_sums.argmax()
    
    # if len(change_points) == 0:
    #     print("no significant change points detected")
    #     return
    end_idx = change_point + 1
    start_idx = end_idx - ref_num_frames
    new_expression_parameters = expression_parameters[start_idx : end_idx]  
      
    # import pdb;pdb.set_trace()
    ## usually at the last frame media player abruptly changes back to the first frame
    ## find the first significant change point
    if (start_idx) < 0: # in case the change at starting point is bigger than the change at last frame
        # import pdb;pdb.set_trace()
        end_idx = change_point + ref_num_frames
        start_idx = change_point + 1
        new_expression_parameters = expression_parameters[start_idx : end_idx + 1]    
    
    np.save(output_path, new_expression_parameters)
    
    print(f"{os.path.basename(reference_video_path)} >> strat index : {start_idx} / shape of trimmed expression parameter : f{new_expression_parameters.shape}")
    
import subprocess
def mux_audio_video(audio_path, video_path, output_path):
    cmd = f"ffmpeg -y -i {audio_path} -i {video_path} -c:v copy -map 0:a -map 1:v -c:a aac {output_path}"
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def concat_videos(vid1_paths : list, output_path : str, row_wise=True):
    
    # load all videos
    clips = [mp.VideoFileClip(video) for video in vid1_paths]
    
    # ensure all clips have the same height
    min_height = min(clip.h for clip in clips)
    clips = [clip.resize(height=min_height) for clip in clips]
    
    if row_wise:
        # concat videos row wise 
        final_clip = mp.clips_array([clips])
    else:
        # concat videos column wise
        final_clip = mp.clips_array([[clip] for clip in clips])
    
    final_clip.write_videofile(output_path, codec = "libx264", fps=30)
    
    return 


def bshp_2_vtx(bhsp_anim_seq : np.array = None, face_model = None, id_param : np.array = None):
    if face_model == None:
        face_model = face_model_io.load_face_model('/input/inyup/ICT-FaceKit/FaceXModel')

    vertex_animation = []
    face_model.set_identity(id_param)
    face_model.deform_mesh()

    for frame_idx, parameter in enumerate(bhsp_anim_seq):
        # # blendshape 53
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        #### TODO ####
        vs = face_model._deformed_vertices.copy() # need to flatten this and only extract full face area [0:9408]
        vertex_animation.append(vs)
    
    vertex_animation = np.array(vertex_animation)
    # import pdb;pdb.set_trace()
    vertex_animation = vertex_animation[:,:9409,:].reshape(-1,28227)
    
    return vertex_animation

def gaussian_kernel1d(kernel_size=5, sigma=1.0):
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel
    
def apply_gaussian_filter(tensor, kernel_size=5, sigma=1.0):
    """Applies a Gaussian filter to the tensor.
    Args:
        tensor (torch.tensor): input tensor
        kernel_size (int): size of the kernel
        sigma (float): sigma value for gaussian filter
    """
    kernel_size = int(kernel_size)

    # Generate the 1D Gaussian kernel
    kernel = gaussian_kernel1d(kernel_size, sigma)
    kernel = kernel.reshape(1, 1, -1) # (out_channels, in_channels, kernel_size)
    kernel = kernel.repeat(tensor.size(1), 1, 1) # [128, 1, kernel_size]
    tensor = tensor.transpose(0, 1).unsqueeze(0) # [1, 128, T]

    filtered_tensor = F.conv1d(tensor, kernel, padding=(kernel_size // 2), groups=tensor.size(1))

    # Transpose back to original shape
    filtered_tensor = filtered_tensor.squeeze(0).transpose(0, 1)
    return filtered_tensor

def np_gaussian_kernel1d(kernel_size=5, sigma=1.0):
        x = np.arange(kernel_size).astype(float) - (kernel_size - 1) / 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

from scipy.signal import convolve 
def np_apply_gaussian_filter(array : np.array, kernel_size=5, sigma=1.0):
    """Applies a Gaussian filter to the tensor.
    Args:
        tensor (np.array): input numpy array
        kernel_size (int): size of the kernel
        sigma (float): sigma value for gaussian filter
    """
    kernel_size = int(kernel_size)
    pad_width = kernel_size // 2
    
    padded_array = np.pad(array, ((pad_width, pad_width), (0,0)), mode='reflect') # (0,0) means no padding applied to the second axis, which will be vertex dimension
    # Generate the 1D Gaussian kernel
    kernel = np_gaussian_kernel1d(kernel_size, sigma)
    ## Apply the convoltuion along the last axis (assuming 2D array: channels, time)
    # filtered_array = convolve(array, kernel[:, np.newaxis], mode='same', method='direct')
    ## Apply the convoltuion along the last axis (assuming 1D array)
    # import pdb;pdb.set_trace()
    if array.shape[1] == 1:
        filtered_array = convolve(padded_array, kernel, mode='same', method='direct')
    else: 
        filtered_array = np.apply_along_axis(lambda m: convolve(m, kernel, mode='same', method='direct'), axis=0, arr=padded_array)

    return filtered_array

from scipy.ndimage import gaussian_filter1d as gf
def new_np_gaussian_kernel1d(array: np.ndarray, sigma=1.0):
    return gf(array, sigma=sigma, axis=0, mode='reflect')

###############
## render utils

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, no_parsing=False):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    # import pdb;pdb.set_trace()
    V = V @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    if no_parsing:
        return V
    VF = V[F]
    return VF

def calc_face_norm(vertices, faces, mode='faces'):
    """
    Args
        vertices (np.ndarray): vertices
        faces (np.ndarray): face indices
    """

    fv = vertices[faces]
    span = fv[:, 1:, :] - fv[:, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)
    
    if mode=='faces':
        return norm
    
    # Compute mean vertex normals manually
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += norm[i]

    # Normalize the vertex normals
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)

    return norm_v

