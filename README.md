# natural-language-based-3d-facial-animation-generation
generates facial expression via ICT-FaceKit using gpt api 

📥 Installation

This repository uses Git submodules (e.g., for ICT-FaceKit).
Please follow the steps below to correctly clone and set up the project.

1. Clone the repository

git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

Replace <YOUR_USERNAME> and <YOUR_REPO_NAME> with your actual GitHub values.

2. Initialize and update submodules

This repository includes external dependencies managed as Git submodules.
Run the following command to download all submodule contents:

git submodule update --init --recursive

If you skip this step, submodule folders (e.g., ICT-FaceKit/) may appear empty or incomplete.

3. (Optional) Update submodules to the latest version

If you want to pull the latest commits from each submodule:

git submodule update --remote --recursive
