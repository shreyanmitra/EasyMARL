# EasyMARL

Single-file MARL implementations with sample, customizable, <u>discrete</u> grid world environments. Adapted from natashamjacques/multigrid.

**NOTE: This project is in progress. Source code is not stable and unintended behavior might emerge while running. Please notify us of issues so that we can fix them**

## Currently Supported Algorithms
(Help us by adding some!)

1. IPPO (Independent Proximal Policy Optimization)


## Metacontroller
The metacontroller (housed at multiagent_metacontroller.py) is the middle man between the agents and the MARL environment. While we tried to keep its implementation as generic as possible, it might need to be updates as new reinforcement learning methods are added to agent.py

## Quickstart
To run, clone this repo and type the following on the command line:
```
python main.py <arguments>
```

Available command line arguments are ``env_name``, ``mode``, ``with_expert``, ``debug``, ``seed``, ``keep_training``, ``visualize``, ``video_dir``, ``load_checkpoint_from``, and ``wandb_project``

A Sphinx documentation of the source code has not been created yet. We will update the README with a link to the readthedocs when completed.

## GUI
The user interface provides a way to train and visualize agents without any code.
To run, clone this repo and type the following on the command line:

```
python gui.py
```


## Contribute
Feel free to add new implementations or fix any issues by filing a pull request.
