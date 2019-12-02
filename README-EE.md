# AIM402 Deep learning with PyTorch

## Build, Test, and Tune Machine Learning Models with PyTorch & Amazon SageMaker

This set of labs aim to enable to develop end-to-end NLP solutions with [PyTorch](pytorch.org) using  [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 
for the full Machine Learning Lifecycle. Each lab is independent, but during the presentation we would go sequentially. 
We wish to thank the authors and code committers of projects listed in the reference section, the labs would not be possible without this 
excellent material. 

It takes around 2 hours to complete these set of labs

## Instructions

Follow the instructions below to start the workshop:

1. Open the Amazon SageMaker console at https://console.aws.amazon.com/sagemaker/.
1. Change region to US East (N. Virginia)
1. Choose Notebook instances, then choose Create notebook instance.
On the Create notebook instance page, provide the following information (if a field is not mentioned, leave the default values):
    1. For Notebook instance name, type a name for your notebook instance.
    1. For Instance type, choose ml.p2.xlarge.
    For IAM role, choose Create a new role, then choose Create role. Make sure to say 'Any s3 buckets'
    1. Choose Create notebook instance.
    1. In a few minutes, Amazon SageMaker launches an ML compute instance—in this case, a notebook instance—and attaches an ML storage volume to it. 
    The notebook instance has a preconfigured Jupyter notebook server and a set of Anaconda libraries.
1. Wait for the notebook to launch, then click on 'Open JupyterLab'
1. Once JupyterLab is launched, Goto File->New->Terminal
   1. Once in the terminal do - `cd SageMaker/`
   1. And clone the workshop git in this notebook - `git clone https://github.com/C24IO/Reinvent2019Pytorch.git`
   1. The lab contents should appear in the left hand side navigation bar, please proceed working through the instructions.
   
### Please note: If asked to select a kernel - please select - conda_pytorch_36 everytime.
   
_version Nov 29 2019_