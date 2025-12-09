import sys
if __name__ != "main":
    sys.path.append("/source/inyup/IEFA/src/FaceMEO/llm/")
from motion import FacialMotion


class Motion_DB:
    def __init__(self):
        self.motion_database = {}   # {motion_name: {keyframes, curve_type}}
        self.motion_history = []    # 저장된 motion_name 순서대로 저장 (Undo용)

    def save_motion(self, motion: FacialMotion, motion_name: str):
        """
        Save the current motion state.
        """
        self.motion_database[motion_name] = {
            "keyframes": {frame: value.copy() for frame, value in motion.keyframes.items()},
            "curve_type": motion.curve_type
        }
        self.motion_history.append(motion_name)
        print(f"[Motion_DB] Saved '{motion_name}' successfully.")


    def load_motion(self, motion_name: str, return_dict=True):
        """
        Load a motion by name.
        """
        motion_info = self.motion_database.get(motion_name)
        if motion_info is None:
            raise ValueError(f"[Motion_DB] Motion '{motion_name}' not found.")
        
        if return_dict:
            return motion_info
        else:
            return FacialMotion(motion_info)


    def undo_last_motion(self):
        """
        Undo: Return a copy of the previous motion without modifying the DB.
        Does NOT remove any motion from the history.
        """
        if len(self.motion_history) < 2:
            raise ValueError("[Motion_DB] Cannot undo: no previous motion to revert to.")
        
        prev_motion_name = self.motion_history[-2]
        motion_info = self.motion_database.get(prev_motion_name)

        print(f"[Motion_DB] Undo: loading keyframes from '{prev_motion_name}', but preserving history.")

        return motion_info


    def revert_to_motion(self, motion_name: str):
        """
        Revert to a specific motion snapshot.
        Does NOT trim motion history.
        Just returns the selected motion’s data.
        """
        if motion_name not in self.motion_database:
            raise ValueError(f"[Motion_DB] Motion '{motion_name}' does not exist for revert.")
        
        motion_info = self.motion_database.get(motion_name)
        print(f"[Motion_DB] Reverted to snapshot '{motion_name}' without trimming history.")
        
        return motion_info
    
    
    def get_latest_motion_name(self):
        """
        Returns the latest saved motion name.
        """
        if not self.motion_history:
            return None
        return self.motion_history[-1]

