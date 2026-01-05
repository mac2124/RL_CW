# Run Instructions

## Environment Setup

### 1. Operating System Requirements
Ensure you are running one of the following operating systems:
* **Windows:** Windows 10 or 11 (via **WSL2**)
* **Linux:** (manylinux1) Ubuntu 22.04 is recommended
* **macOS:** 10.13 (High Sierra) or 10.14 (Mojave)

### 2. Installation Guides
* **Windows (WSL2):** Follow this [YouTube Setup Guide](https://www.youtube.com/watch?v=vPnJiUR21Og).
* **Linux:** Follow the [Stable-Retro Linux Docs](https://github.com/Farama-Foundation/stable-retro/blob/master/docs/linux_installation.md).
* **macOS:** Follow the [Stable-Retro macOS Docs](https://github.com/Farama-Foundation/stable-retro/blob/master/docs/macos_installation.md).

### 3. Python & Dependencies
**Supported Python Versions:** 3.7 to 3.12

> **Note:** Using a virtual environment is highly recommended.

We followed the official instructions from [Farama-Foundation/stable-retro](https://github.com/Farama-Foundation/stable-retro) for installing all dependencies and setting up the virtual environment.

## ROM Setup
You must provide the game ROM yourself. We used the version from **[Vimm's Lair](https://vimm.net/vault/2180)**.

1. Download the ROM file (`.md` or `.bin`).
2. Place the file inside a folder within the project directory.
3. Run the import command:
   ```bash
   python3 -m retro.import .

### Running The Code

The code is run using the command

    
    python3 MK2_env_setup.py

**Note:** if there is an error finding the game you may have to run the following:


    python3 MK2_env_setup.py --game "insert exact game name here"


To switch between Training, Resuming, and Watching, you must modify the if __name__ == "__main__": block at the bottom of the script.

1. Training a new agent
    ```Python

    if __name__ == "__main__":
        main()
        # resume_training()
        # watch_agent_play()
2. Resume Training
    ```Python

    if __name__ == "__main__":
        # main()
        resume_training()
        # watch_agent_play()
3. Watch Agent Play
    ```Python

    if __name__ == "__main__":
        # main()
        # resume_training()
        watch_agent_play()


