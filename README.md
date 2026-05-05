# Assignment
This repository has the code we used for membership inference attack. The script loads the given public dataset, private dataset, and target model then it computes membership scores using loss/confidence-based features and shadow-model LiRA-style scoring and generates `submission.csv` file.

##Files
mia.sub - HTCondor submit file to run the job
submission.csv - Generated output file
run.sh - Shell script file
task_template.py - Contains the code

##Input Files
pub.pt
priv.pt
model.pt

How to Setup and Run:

1. Used sudo openconnect vpn.hiz-saarland.de command to connect to university VPN.
2. Then in VS Code created a Connected to SSH remote host using conduit.hpc.uni-saarland.de or conduit2.hpc.uni
saarland.de.
3. Provided the cluster password to successfully connect to host.
4. Then used the commands mkdir ~/tml26_task1 then cd ~/tml26_task1.
5. Then downloaded the following files,
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"  
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"  
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/task_template.py"
6. In task_template.py file added the API key.
7. Then we ran the command mkdir -p runlogs once.
8. Thereafter, for running the code and submitting used the command condor_submit mia.sub
9. For checking the job,  condor_q was used.
