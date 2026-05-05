# Assignment
This repository has the code we used for membership inference attack. The script loads the given public dataset, private dataset, and target model then it computes membership scores using loss/confidence-based features and shadow-model LiRA-style scoring and generates `submission.csv` file.

## Files
- README.md - explains how to recreate the best result
- task_template.py - Contains the script


How to Setup and Run:

1. Used `sudo openconnect vpn.hiz-saarland.de` command to connect to university VPN.
2. Then in VS Code created a Connection to SSH remote host using `conduit.hpc.uni-saarland.de or conduit2.hpc.uni
saarland.de`.
3. Then we used the cluster password provided to successfully connect to host.
4. Then used the commands `mkdir ~/tml26_task1` then `cd ~/tml26_task1`.
5. Then downloaded the following files,
``wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"``  
``wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"``   
``wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"`` 
``wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/task_template.py"``
6. In task_template.py file added the API key provided.(Note: API key has been removed from the public submission, Please replace with your own valid API Key).
7. Then we run the command `mkdir -p runlogs` once.
8. Thereafter, for running the code and submitting used the command `condor_submit mia.sub`.
9. To check whether the job is running command `condor_q` was used.
10. To visit the leaderboard, this link was used `http://34.63.153.158/leaderboard_page`. 
