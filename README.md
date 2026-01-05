Run Instructions

Environment setup:

OS options:
Windows 10, 11 (via WSL2)
macOS 10.13 (High Sierra), 10.14 (Mojave)
Linux (manylinux1). Ubuntu 22.04 is recommended

for WSL2 follow: https://www.youtube.com/watch?v=vPnJiUR21Og
for Linux installation follow: https://github.com/Farama-Foundation/stable-retro/blob/master/docs/linux_installation.md
for macOS follow: https://github.com/Farama-Foundation/stable-retro/blob/master/docs/macos_installation.md

Python: 3.7 to 3.12 is supported

We followed instructions from https://github.com/Farama-Foundation/stable-retro for installing all dependencies
and virtual environment setup.

Using a virtual environment is highly recommended

We had to import Mortal Kombat 2 ROM ourselves from here (https://vimm.net/vault/2180). You must put the rom file (.md or .bin) inside a folder and run (python3 -m retro.import).

How to run the code:

The code is run from MK2_env_setup.py (python3 MK2_env_setup.py). In order to train an agent from scratch you must run the main() function
if __name__ == "__main__":
    main() 
    # resume_training()
    # watch_agent_play()

To continue training an agent run the resume_training() function
if __name__ == "__main__":
    # main() 
    resume_training()
    # watch_agent_play()

and to watch an agent play without training run watch_agent_play()
if __name__ == "__main__":
    # main() 
    # resume_training()
    watch_agent_play()

These functions should be called within the if __name__ == "main" code block. After training is complete the model weights and reward scaling stats are saved in the same directory as the script (e.g. ppo_mk3.pt and vec_normalise.pkl).
