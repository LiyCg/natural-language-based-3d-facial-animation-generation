# Natural Language–Based 3D Facial Animation Generation

This repository provides a framework for generating **3D facial animation sequences controlled by natural language instructions**. The system processes high-level semantic commands (e.g., “make the character smile at frame 20 and blink at frame 60”) and automatically produces keyframe-based facial motions using a structured Motion API.

The project includes:
- A Large Language Model (LLM)-based motion generation module  
- A Motion Database (Motion_DB) for maintaining edit history  
- FacialMotion, a blendshape-based animation engine  
- Integration with **ICT-FaceKit** (included as a Git submodule) for 3D mesh decoding and rendering  
 
---

# Features

- **Natural Language → Facial Motion Conversion**  
  Automatically converts user instructions into executable motion-editing code.

- **Blendshape-Based Editing**  
  Generates and adjusts expression keyframes using a structured motion API.

- **Motion History & Undo System**  
  Full undo/revert mechanism for iterative editing.

- **3D Mesh Reconstruction**  
  Converts blendshape animations into vertex animations through ICT-FaceKit utilities.

- **Modular Architecture**  
  Motion generation, decoding, and rendering are separated for flexibility.
 
---

# Installation

This repository uses **Git submodules** (e.g., ICT-FaceKit).  
Follow the steps below to properly install and configure the project.


## 0. Install Dependencies
Using pip:
```bash
pip install -r requirements.txt
```

Using Conda:
```bash
conda env create -f environment.yml
conda activate <env_name>
```


## 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```


## 2. Initialize Submodules
Submodules are not automatically pulled during a standard clone. Run:
```bash
git submodule update --init --recursive
```
If skipped, folders such as ICT-FaceKit/ may appear empty.

---

# Licence

This project is licensed under the AGPL-3.0 License.

In summary:

You may use, modify, and distribute this software.

Any modified version or derivative work must also be released under AGPL-3.0.

If you run this software as part of a network service, you must provide users access to the full corresponding source code, including your modifications.

Commercial use is allowed, but the copyleft requirement still applies.

See the full license text in:

LICENSE

---

# Contact

For questions or collaboration inquiries:

Inyup Lee
📧 leeinyup123@kaist.ac.kr

KAIST Visual Media Lab