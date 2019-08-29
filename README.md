
## Folders: 

### Src  
This is the base folder for the code.  hyperparameter_comparison_pipeline.py is the main file that produces a summary with one model, and produces a reconstruction with another model.  results_script.py generates the results files--see the instructions below for running it.  generate_plots.py generates plots from the results files.  You need to first run the results script before plotting.  

### Extraction methods  
This folder contains the summarization methods for different models of humans.  active_learning.py is the summary extraction method for imitation learning, and machine_teaching.py is the summary extraction method for inverse RL.  
 
### Models
This folder contains the models of humans for reconstruction of the policy.  GRF.py is the imitation learning model, and irl_maxent.py is the inverse RL model.  vaue_iteration.py is called in irl_maxent.py.   

### Simulators  
This folder contains the simulator code for the domains and a base simulator class.  For now, it only contains the random gridworld because we adapted other peoples' code for the other 2 domains and I wasn't sure whether we should re-publish our version.

## Running Instructions

### Commands for Results  
You'll need to run both of these commands before plotting.  The first one generates the hyper parameter search results (figure 3 in the ArXiv version: https://arxiv.org/pdf/1905.13271.pdf).  Expect this to take around 30 minutes.  The second one generates the results for figure 1 in the paper.  Expect this to take around 5 minutes.  You can replace -r 0 in either command with -r n to save the nth random restart.  We use 75 random restarts in the paper.  
python results_script.py -d gridworld -r 0  
python results_script.py -d gridworld -c -r 0  

To generate the figures, first open generate_plots.py and change n_runs = 1 (line 267) to however many random restarts you've generated, then run generate_plots.py.

### Dependencies:  
numpy, scipy, scikit-learn, matplotlib/pylab, threading, tqdm, pickle, itertools, copy, abc, subprocess, argparse  

## Contact Info

You can email me at isaaclage@g.harvard.edu if you have any questions or anything comes up while you're trying to run the code.


