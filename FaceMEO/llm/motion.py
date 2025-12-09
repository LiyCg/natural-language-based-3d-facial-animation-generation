import numpy as np
## These are the identifiers you can use. You can ONLY use these identifiers for guessing what identifiers should be used from edit instruction.
emotions = {
    "angry": [
        'brow_lowerer',       # AU04: Lower the brows deeply for a furrowed look
        'outer_brow_raiser',  # AU02: Slightly raise the outer brows for added strain
        'lid_tightener',      # AU07: Tighten the eyelids
        'nose_wrinkler',      # AU09: Wrinkle the nose to add tension
        'lip_stretcher',      # AU20: Stretch the lips wide
        # 'mouth_stretch',      # AU27: Open the mouth aggressively
        'chin_raiser',        # AU17: Tense the chin
        'lip_presser'         # AU24: Press the lips together for added mouth tension
    ],
    "contempt" : ['dimpler', 'lip_corner_puller', 'cheek_raiser'],
    "disgusted" : ['nose_wrinkler', 'upper_lip_raiser', 'lip_corner_depressor'],
    "fear" : ['inner_brow_raiser', 'outer_brow_raiser', 'upper_lid_raiser', 'lid_tightener', 'mouth_stretch'],
    "happy" : ['lip_corner_puller','cheek_raiser','upper_lid_raiser'],
    "sad" : ['inner_brow_raiser', 'brow_lowerer', 'lip_corner_depressor', 'chin_raiser'],
    "surprised" : ['inner_brow_raiser', 'outer_brow_raiser','upper_lid_raiser','mouth_stretch']
}
FACS_units = {
    'inner_brow_raiser': ['brow_inner_up_l', 'brow_inner_up_r'],            # AU01
    'outer_brow_raiser': ['brow_outer_up_l', 'brow_outer_up_r'],            # AU02
    'brow_lowerer': ['brow_down_l', 'brow_down_r'],                         # AU04
    'cheek_raiser': ['cheek_squint_l', 'cheek_squint_r'],                   # AU06
    'upper_lid_raiser': ['eye_wide_l', 'eye_wide_r'],                       # AU05
    'lid_tightener': ['eye_squint_l', 'eye_squint_r'],                      # AU07
    'nose_wrinkler': ['nose_sneer_l', 'nose_sneer_r'],                      # AU09
    'upper_lip_raiser': ['mouth_shrug_upper'],                              # AU10
    'nasolabial_deepener': ['mouth_upper_up_l', 'mouth_upper_up_r'],        # AU11
    'lip_corner_puller': ['mouth_smile_l', 'mouth_smile_r'],                # AU12
    'dimpler': ['mouth_dimple_l', 'mouth_dimple_r'],                        # AU14
    'lip_corner_depressor': ['mouth_frown_l', 'mouth_frown_r'],             # AU15
    'lower_lip_depressor': ['mouth_lower_down_l', 'mouth_lower_down_r'],    # AU16
    'chin_raiser': ['mouth_shrug_lower'],                                   # AU17
    'lip_pucker': ['mouth_pucker'],                                         # AU18
    'lip_stretcher': ['mouth_stretch_l', 'mouth_stretch_r'],                # AU20
    'lip_funneler': ['mouth_funnel'],                                       # AU22
    'lip_presser': ['mouth_press_l', 'mouth_press_r'],                      # AU24
    'mouth_stretch': ['jaw_open'],                                          # AU27
    'lip_suck': ['mouth_roll_lower', 'mouth_roll_upper'],                   # AU28
    'jaw_thrust': ['jaw_forward'],                                          # AU29
    'jaw_sideways_left': ['jaw_left'],                                      # AU30 (left)
    'jaw_sideways_right': ['jaw_right'],                                    # AU30 (right)
    'cheek_blow': ['cheek_puff_l', 'cheek_puff_r'],                         # AU33
    'blink': ['eye_blink_l', 'eye_blink_r'],                                # AU45
    'eyes_turn_left': ['eye_look_out_l', 'eye_look_in_r'],                  # AU61
    'eyes_turn_right': ['eye_look_out_r', 'eye_look_in_l'],                 # AU62
    'eyes_up': ['eye_look_up_l', 'eye_look_up_r'],                          # AU63
    'eyes_down': ['eye_look_down_l', 'eye_look_down_r']                     # AU64
}
full_blendshape_targets = [
    'brow_down_l',         # browDown_L # overwritten by both down
    'brow_down_r',         # browDown_R
    'brow_inner_up_l',     # browInnerUp_L
    'brow_inner_up_r',     # browInnerUp_R
    'brow_outer_up_l',     # browOuterUp_L
    'brow_outer_up_r',     # browOuterUp_R
    'cheek_puff_l',        # cheekPuff_L
    'cheek_puff_r',        # cheekPuff_R
    'cheek_squint_l',      # cheekSquint_L
    'cheek_squint_r',      # cheekSquint_R
    'eye_blink_l',         # eyeBlink_L
    'eye_blink_r',         # eyeBlink_R
    'eye_look_down_l',     # eyeLookDown_L
    'eye_look_down_r',     # eyeLookDown_R
    'eye_look_in_l',       # eyeLookIn_L
    'eye_look_in_r',       # eyeLookIn_R
    'eye_look_out_l',      # eyeLookOut_L # overwritten by both eye moving
    'eye_look_out_r',      # eyeLookOut_R # 여기까지 확인됨
    'eye_look_up_l',       # eyeLookUp_L
    'eye_look_up_r',       # eyeLookUp_R
    'eye_squint_l',        # eyeSquint_L
    'eye_squint_r',        # eyeSquint_R
    'eye_wide_l',          # eyeWide_L -> for eyes at max 
    'eye_wide_r',          # eyeWide_R -> for eyes at max
    'jaw_forward',         # jawForward
    'jaw_left',            # jawLeft
    'jaw_open',            # jawOpen -> for mouth at max
    'jaw_right',           # jawRight
    'mouth_close',         # mouthClose
    'mouth_dimple_l',      # mouthDimple_L # overwritten by both dimpling
    'mouth_dimple_r',      # mouthDimple_R
    'mouth_frown_l',       # mouthFrown_L -> 입꼬리 at min
    'mouth_frown_r',       # mouthFrown_R -> 입꼬리 at min
    'mouth_funnel',        # mouthFunnel
    'mouth_left',          # mouthLeft
    'mouth_lower_down_l',  # mouthLowerDown_L
    'mouth_lower_down_r',  # mouthLowerDown_R
    'mouth_press_l',       # mouthPress_L
    'mouth_press_r',       # mouthPress_R
    'mouth_pucker',        # mouthPucker
    'mouth_right',         # mouthRight
    'mouth_roll_lower',    # mouthRollLower
    'mouth_roll_upper',    # mouthRollUpper
    'mouth_shrug_lower',   # mouthShrugLower
    'mouth_shrug_upper',   # mouthShrugUpper
    'mouth_smile_l',       # mouthSmile_L -> 입꼬리 at max
    'mouth_smile_r',       # mouthSmile_R -> 입꼬리 at max
    'mouth_stretch_l',     # mouthStretch_L 
    'mouth_stretch_r',     # mouthStretch_R
    'mouth_upper_up_l',    # mouthUpperUp_L
    'mouth_upper_up_r',    # mouthUpperUp_R
    'nose_sneer_l',        # noseSneer_L
    'nose_sneer_r'         # noseSneer_R
]

## This is the FacialMotion class.
import numpy as np
class FacialMotion:
    def __init__(self, motion=None, total_frames=100):
        self.total_frames = total_frames
        self.keyframes = {}  # {frame: np.ndarray(53)}
        self.default_value = np.zeros(53)
        self.curve_type = "smoothstep"
        self.output_animation_seq = np.zeros((self.total_frames, 53), dtype=np.float32)  # <= 추가!

        if isinstance(motion, dict):
            raw_keyframes = motion.get("keyframes", {})
            self.keyframes = {int(f): np.array(v) for f, v in raw_keyframes.items()}
            self.curve_type = motion.get("curve_type", "smoothstep")
            self.output_animation_seq = self.interpolate()  # 바로 interpolate 결과 저장
           
    def add_keyframe(self, frame: int, identifiers: list = None, intensity: list = None):
        """
        Add a keyframe at a specific frame.
        If identifiers is None, it creates a neutral (zero) keyframe.
        """
        assert 0 <= frame < self.total_frames, f"Frame {frame} out of range."
        param = np.zeros(53)
        if identifiers:
            assert intensity is not None, "Intensity must be provided if identifiers are given."
            for ident, inten in zip(identifiers, intensity):
                indices = self._identifier_to_indices([ident])
                for idx in indices:
                    param[idx] += inten 

        self.keyframes[frame] = param


    def adjust_keyframe(self, frame: int, identifiers: list, deltas: list):
        """
        Adjust existing keyframe by adding/subtracting deltas to specified identifiers.
        """
        assert frame in self.keyframes, f"No keyframe at frame {frame}."
        assert len(identifiers) == len(deltas), "Identifiers and deltas must match."
        for ident, delta in zip(identifiers, deltas):
            indices = self._identifier_to_indices([ident])
            for idx in indices:
                self.keyframes[frame][idx] += delta
                self.keyframes[frame][idx] = max(0.0, self.keyframes[frame][idx])  # Clamp at 0


    def interpolate(self):
        output = np.zeros((self.total_frames, 53), dtype=np.float32)

        if not self.keyframes:
            return output

        sorted_frames = sorted(self.keyframes.keys())

        if len(sorted_frames) == 1:
            frame = sorted_frames[0]
            value = self.keyframes[frame]
            for f in range(self.total_frames):
                output[f] = value
            self.output_animation_seq = output
            return output

        for i in range(len(sorted_frames) - 1):
            start = sorted_frames[i]
            end = sorted_frames[i + 1]
            start_val = self.keyframes[start]
            end_val = self.keyframes[end]

            for f in range(start, end + 1):
                t = (f - start) / (end - start)
                t = self._apply_curve(t)
                output[f] = (1 - t) * start_val + t * end_val

        # Before first keyframe: hold
        first_key = sorted_frames[0]
        for f in range(0, first_key):
            output[f] = self.keyframes[first_key]

        # After last keyframe: hold
        last_key = sorted_frames[-1]
        for f in range(last_key + 1, self.total_frames):
            output[f] = self.keyframes[last_key]

        self.output_animation_seq = output
        return output


    def generate_keyframe(self, frame: int, identifiers: list, intensity: list, delta_mode=False):
        """
        FacialMotion 객체 메서드: keyframe 추가 또는 수정
        """
        assert isinstance(identifiers, list), "identifiers must be a list."
        assert isinstance(intensity, list), "intensity must be a list."
        assert len(identifiers) == len(intensity), f"identifiers ({len(identifiers)}) and intensity ({len(intensity)}) must match."

        print("[generate 전 keyframes]:", sorted(self.keyframes.keys()))

        if not delta_mode:
            self.add_keyframe(frame=frame, identifiers=identifiers, intensity=intensity)
        else:
            if frame not in self.keyframes:
                raise ValueError(f"No existing keyframe at frame {frame} to adjust.")
            self.adjust_keyframe(frame=frame, identifiers=identifiers, deltas=intensity)

        self.interpolate()
        print("[generate 후 keyframes]:", sorted(self.keyframes.keys()))
            

    def _apply_curve(self, t):
        if self.curve_type == "smoothstep":
            return t * t * (3 - 2 * t)
        else:
            return t  # linear fallback


    def _identifier_to_indices(self, identifiers: list):
        
        expanded = []
        for identifier in identifiers:
            if identifier in emotions:
                facs_units = emotions[identifier]
                for facs in facs_units:
                    blendshape_targets = FACS_units[facs]
                    for bt in blendshape_targets:
                        if bt not in expanded:
                            expanded.append(bt)
            elif identifier in FACS_units:
                blendshape_targets = FACS_units[identifier]
                for bt in blendshape_targets:
                    if bt not in expanded:
                        expanded.append(bt)
            elif identifier in full_blendshape_targets:
                if identifier not in expanded:
                    expanded.append(identifier)
            else:
                raise ValueError(f"Identifier '{identifier}' is not valid.")
        
        return [full_blendshape_targets.index(bt) for bt in expanded]


    def save(self):
        """
        Save keyframe and curve info to dictionary.
        """
        return {
            "keyframes": self.keyframes,
            "curve_type": self.curve_type
        }

