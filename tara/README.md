# Time Signal Detection with Neural Networks and Wasserstein Loss
&nbsp;  
### Introduction
This project is an example of time signal detection with neural networks and 1D wasserstein statistical loss. It is created for **Tsinghua-Jinping-PMT-Simulation-Dataset**. Behaviors of multiple Photomultiplier Tubes (PMTs) are simulated and recorded in the dataset.
&nbsp;  
In the experiment set-up, PMT detectors receive photons and generate digitized electronic signals. For each neutrino detecting event, multiple photons are emitted and later received by PMT detectors within a short period of time, generating electronic signal pulses in the read-outs. The key problem of the simulation is to find out the time and possiblity of photon receiptions based on their corresponding electronic signal pulses. Considering the prediction accuracy in both possiblity and time, a proper judgement standard is the **Wasserstein Distance** between the analysis **outputs** and their corresponding **ground truths**.  
&nbsp;  

### DataSets
One can find the dataset and try out with the PlayGround in: https://nu.airelinux.org/ . Unfortunately, the submission channels for the first and the final competitions are currently offline.  
All survice are avaliable for IPv6 visiters: https://data-contest.net9.org/ .  
&nbsp;  

### Cautions for Data Submission
Please be aware that the judgement program has already made a time shift for -5 channels (roughly equal to the time difference between a incoming photon and its electronic pulse peak), in order to simplify the time signal detection problem.     
The training and judgement datasets for the PlayGround are not generated under the same parameter, so a minor additional error may exists to worsen the processing scores.  
**The results are judged by Wasserstein Distance, which means all weight distribution will be normalized.** The absolute quantities of the output is unimportant, because only an additional division by an amplitude factor is needed to make a perfect magnitude match to the Ground Truth.     
We also encourage adding a L1-loss to our wasserstein loss, but this will worsen the Wasserstein Score.  
**Wasserstein Distance Score: The smaller, the better.**

&nbsp;   

### Methods
Based on intrinsic characteristics of the Physics signal, One Dimensional Convolution Neural Networks (1D-CNN) are used as the fundamental blocks building the neural networks.  

In order to balance the trade-off relationship between the calculation speed and the prediction accuracy, 4-layers of the 1D CNN is used to build up the deep neural networks. Deeper neural network may give rise to the accuracy, but will also lower the speed in training and result generation processes.

The Neural Networks is trained with self-designed Wasserstein Loss. For more information, please go to: https://github.com/TakaraResearch/Pytorch-1D-Wasserstein-Statistical-Loss or search **Pytorch 1D Wasserstein Loss** on Github.

Further explorations are encouraged.  

&nbsp;    

### Documents
|file | Introduction | Explaination |
|---|---|---|
|Data_Pre-Processing.py:       |Training Data Pre-Processing | Transfer .h5 data into numpy vectors and save as .npz files  
|Data_Processing.py:           |Neural Network training for the first round | train, update and save network parameters  
|Continue_Training.py:         |Continue Neural Network training | load, train, update and save network parameters
|Prediction_Pre-Processing.py: |Pre-Processing for data to be "Predicted" | Transfer .h5 data into numpy vectors and save as .npz
|Prediction_Processing_Total.py|Make "Prediction" based on the trained network model| Generate photon result and transfer it into a .h5 file

&nbsp;  

### Results
Our model has balanced the result generation speed and the training accuracy. Till now (2019.07), we have submitted the best result in all the three evaluation tests. (Submissions later than Jun.02 are not ranked on the leaderboard for the First and Final Contest, since the contest has ended. )

| Test | Generation Time | Wasserstein Score|2nd Best |
|---|---|---|---|
|PlayGround | ~1min 1CPU | 1.137 | 2.031 |
|First Contest | ~20min 3CPU | 0.965 | 1.396 |
|Final Contest | ~20min 3CPU | 0.852 | 1.219 |

CPU: Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz   
*If more CPU kernels or GPU kernels are used, the training and generation speed will be significantly higher.*


