## These are more examples you can refer to for appropriate generation of code within our systemp expectation. 

from motion_io import Motion_DB
from motion import FacialMotion

# initialize motion DB
db = Motion_DB()
# Add a slight smiling expression at front, making the smile reach its peak in the middle of the sequence.
def slight_smile_peak_middle():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person is talking. 
    # the desired edit is to add a slight smiling expression at the front, making the smile reach its peak in the middle of the sequence.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'reach its peak in the middle' indicates the top expression should be in the middle. So insert point is at mid sequence
    insert_point = int(motion_1.seq_len // 2)
    motion_1.set_insert_point(insert_point)
    # involved identifiers are "happy"
    identifiers = ["lip_corner_puller"]
    # 'peak' indicates highest in intensity, so the default intensity
    motion_1.activate_hierarchical(identifiers=identifiers)
    # 'making the smile reach its peak in the middle of the sequence' indicates the middle of the sequence for front keyframe speed, so seq_len/2, replacing so replace=True
    speed_locations = ["front"]
    speed = [int(motion_1.seq_len // 2)]
    motion_1.set_speed(speed_locations, speed, replace=True)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# the person is talking. Make a face when someone won a lottery all along the sequence. 
def face_win_lottery():
    # initialize motion(only once for saving "motion_1")
    motion_1 = FacialMotion() # loads 'FacialMotion' class instance, the first one is original speech animation motion 

    # the original motion is that the persion talking. 
    # the desired edit is to ativate surprised face in the middle of the motion

    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True 
    # 'all along the sequence' indicates 0 in 'insert_point', so default
    # the involved identifiers are 'surprised' and 'happy'
    # no specification on intensity, so default 
    identifiers = ["surprised", "happy"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    # 'all along the sequence' also indicates the entire length for keyframe speed, so seq_len
    speed_locations = ["keyframes"]
    speed = [motion_1.seq_len]
    motion_1.set_speed(speed_locations, speed, replace=True) # replace, not multiplication
    
    motion_1.activate_blendshape() # retrieves all set motion and converts into expression code to finally deconde into facial animation 
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a brief cheek puff at frame 35. 
def cheeck_puff():
    # initialize motion
    motion_1 = FacialMotion() 
    
    # the original motion is that the persion talking. 
    # the desired edit is to ativate cheek puff at frame 35 

    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True 
    # 'at frame' literally means at frame "35"
    insert_point = 35
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 'cheek_blow' 
    # no specification on intensity, so default  
    identifiers = ["cheek_blow"]
    motion_1.activate_hierarchical(identifiers=identifiers) 
    # 'brief' indicates short in keyframe speed, so just half the default(=15) 
    speed_locations = ["keyframes"] 
    speed = [0.5] 
    motion_1.set_speed(speed_locations, speed, replace=False)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a sharp brow furrow and narrow the eyes starting from almost the end of the motion (to show deep concentration or seriousness). 
def brow_furrow_eye_narrow():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the persion talking. 
    # the desired edit is to ativate eye squint face at almost the end
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'almost the end' indicates roughly 0.7 in 'insert_point'
    insert_point = 0.7
    motion_1.set_insert_point(insert_point) 
    # the involved identifiers are 
    # brow_lowerer for brow furrow
    # lid_tightener for narrowed eyes
    # sharp usually means high in intensity, so slighly above 1.0, like 1.2 
    intensities = [1.2, 1.2]
    identifiers = ["brow_lowerer","lid_tightener"]
    motion_1.activate_hierarchical(identifiers, intensities)   
    # no specification on speed_location, so default
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. The person begins to smile with only left corner of mouth. 
def smile_left():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the persion talking. 
    # The desired edit is to ativate smile only with left corner of mouth starting from the motion
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'begins to' indicates 0  in 'insert_point'
    insert_point = 0
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 
    # mouth_smile_l
    # no specification on intensity, so default
    identifiers = ["mouth_smile_l"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")
    
# initialize motion DB
db = Motion_DB()
# The person is talking. Close eyes starting from the middle to the end. 
def close_eyes():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the persion talking. 
    # The desired edit is to close eyes starting from the middle to the end
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'starting from the middle to the end' idicates 0.5 in 'insert_point', 
    insert_point = 0.5
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 
    # blink 
    # 'close eyes' usually indicate total closure of both eyes, so intensities to max range, which is 1.0
    identifiers = ["blink"]
    intensities = [1.0]
    motion_1.activate_hierarchical(identifiers=identifiers, intensities=intensities)
    # 'starting from the middle to the end' indicates number of 'keyframes' are from middle to the end
    speed_locations = ["keyframes"]
    speed = [motion_1.seq_len - int(motion_1.seq_len * insert_point)]
    motion_1.set_speed(speed_locations, speed)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# assume there's already initialized motion db
# Make happy face faster.
def smile_faster():
    # load previous motion
    motion_1 = db.load_motion("motion_1") # changing speed of some expression should mean the expression does exist, so assume there's been already a first editing.
    motion_2 = FacialMotion(motion_1)
    
    # the loaded motion is that the person is talking with happy face starting from in middle of the sequence. 
    # the desired edit is to make the happy face faster

    # changing existing keyframe..  
    # existing identifier used. Leave it as it is. 
    motion_2.replace_activated = False
    # no specification on insert_point, so same to previous 
    motion_2.set_insert_point(motion_1.key_idx)
    # set 'speed' to previous first
    # 'faster' indicates shorter number of frames in 'front' frames in 'speed', so half the default
    motion_2.set_speed(list(motion_1.speed.keys()), list(motion_1.speed.values()), replace=True) # replace, not multiplication
    speed_locations = ["front"]
    speed = [0.5]
    motion_2.set_speed(speed_locations, speed)
    
    # everything else stays the same
    motion_2.activate_blendshape()

    # save motion
    db.save_motion(motion_2, "motion_2")

# initialize motion DB
db = Motion_DB()
# Change back to neutral abrutly.
def back_to_neutral_faster():
    # load previous motion
    motion_1 = db.load_motion("motion_1") # changing speed of some expression should mean the expression does exist, so assume there's been already a first editing.
    motion_2 = FacialMotion(motion_1)
    
    # the loaded motion is that the person is talking with happy face starting from in middle of the sequence. 
    # the desired edit is to make the happy face faster
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is.
    motion_2.replace_activated = False
    # no specification on insert_point, so same as previous
    motion_2.set_insert_point(motion_1.key_idx)
    # set 'speed' to previous first 
    # 'change back to neutral abrutly' indicates shorter number of frames in 'back' frames in 'speed', so half the default, multiplying so replace=False
    motion_2.set_speed(list(motion_1.speed.keys()), list(motion_1.speed.values()), replace=True) # replace, not multiplication
    speed_locations = ["back"]
    speed = [0.5]
    motion_2.set_speed(speed_locations, speed, replace=False)
    
    # everything else stays the same
    motion_2.activate_blendshape()

    # save motion
    db.save_motion(motion_2, "motion_2")    

# initialize motion DB
db = Motion_DB()
# When eyes are fully shut, start smirking only on the left side  
def smile_left_when_eyes_shut():
    # load previous motion
    motion_1 = db.load_motion("motion_1")
    motion_2 = FacialMotion(motion_1)
    
    # the loaded motion is that the person is talking with closed eyes from the middle of the sequence. 
    # The desired edit is when eyes are fully shut, start smirking only on the left side

    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_2.replace_activated = True
    # 'when eyes are fully shut' indicates 'max' value for 'eye_blink_r' or 'eye_blink_l' (no specific eye specified)
    point = "eye_blink_r" 
    motion_2.set_insert_point(motion_2.when_bshp("max", point)) 
    # the involved identifiers are 
    # mouth_smile_l
    # no specification on intensity, so default
    identifiers = ["mouth_smile_l"]
    motion_2.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    
    motion_2.activate_blendshape() 
    
    # save motion 
    db.save_motion(motion_2, "motion_2") 

# initialize motion DB
db = Motion_DB()
# The person is talking. As you talk, widen your eyes more.    
def widen_eyes_more():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the persion talking. 
    # The desired edit is as you talk, widen your eyes more.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'As you talk' indicates 0 in 'insert_point', so default,
    # the involved identifiers are 
    # upper_lid_raiser
    # 'widen more' usually mean higher on intensity, so 1.5x the current value, multiplying so replace=False
    identifiers = ["upper_lid_raiser"]
    intensities = [1.5]
    motion_1.activate_hierarchical(identifiers, intensities, replace=False)
    # no specification on speed_location, so default
    # 'As you talk' indicates sequence length for keyframe speed, so seq_len, so replacing not multiplying, replace = True
    speed_locations = ["keyframes"]
    speed = [motion_1.seq_len]
    motion_1.set_speed(speed_locations, speed, replace=True)
    
    motion_1.activate_blendshape() 
    
    # save motion 
    db.save_motion(motion_1, "motion_1") 

# initialize motion DB
db = Motion_DB()
# The person is talking. Switch to a sad expression and pause it for a second in the middle of the sequence.
def sad_expression():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to activate a sad expression during a pause in the middle of the sequence.
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'in the middle of the sequence' indicate the middle of the sequence, so set 'insert_point' to 0.5
    insert_point = 0.5
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 'sad'
    # no specification on intensity, so default
    identifiers = ["sad"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    # 'pause for a second' indicates 30 number of frames in 'keyframes' in 'speed', value is number of frarmes so replacing, replace=True
    speed_locations = ["keyframes"]
    speed = [30]
    motion_1.set_speed(speed_locations, speed, replace=True)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a surprised expression followed by relaxation towards the end of the sequence.
def surprise_then_relax():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to show a surprised expression followed by relaxation towards the end.
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'towards the end' indicates roughly 0.75 in 'insert_point'
    insert_point = 0.75
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 'surprised'
    # no specification on intensity, so default
    identifiers = ["surprised"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a focused gaze by narrowing the eyes at frame 60.
def focused_gaze():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to narrow the eyes for a focused gaze at the start.
        
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True 
    # 'at the start' indicates 0 in 'insert_point'
    insert_point = 60
    motion_1.set_insert_point(insert_point)
    # the involved identifiers are 'lid_tightener'
    # no specification on intensity, so default
    identifiers = ["lid_tightener"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")