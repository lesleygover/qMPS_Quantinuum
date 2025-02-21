# Getting Started with Quantinuum's Quantum Computers

This portal contains all you need to get started using Quantinuum's H-Series Quantum Computers, powered by Honeywell. 

## Organization

This folder is organized as follows.

    ├── docs               <- Documentation for Quantinuum Hardware, including user guides, product data sheets, and API Specification
    │
    ├── notebooks          <- Jupyter notebook examples utilizing Quantinuum Hardware
          |
          └── qtuum        <- Simple python wrapper for the Quantinuum API, Installation requirements
    │
    ├── PCNs               <- Product Change Notifications, notifications on feature releases and upgrades

---

## Download

Download the complete set of examples by clicking the **Download** button on the bottom-left of the folder viewer.

## Setup

### 1. Install Python

Before creating your own quantum circuits and running on Quantinuum's H-Series quantum computers, install python for your operating system. Two options are provided, a direct Python installation or Miniconda.

**Option 1: Python**
* [Python](https://www.python.org/downloads/)
* On Windows, it is recommended to install python from the Microsoft Store. This will automatically update your PATH settings on Windows Command Prompt. Just search *Python* in the app store.

**Option 2: Miniconda**
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create Environment

Creating a python virtual environment provides many benefits for code development, including ensuring you have a stable, reproducible, and portable code base. There are multiple packages available you can use to create virtual environments. You can choose the one you prefer or use one provided here. The examples name the virtual environment `qtuum`, but this can be any name you prefer.

Note that if you're on Windows use `\` rather than `/` in the commands below.

You will be prompted to enter `Y` or `y` to continue installation for some commands.

**Option 1: Python**

Here the package `virtualenvwrapper` is used since it is more user-friendly than the default python `virtualenv` and `venv` packages. For more information on `virtualenvwrapper` see [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) and the [virtualenvwrapper command line reference](https://virtualenvwrapper.readthedocs.io/en/latest/command_ref.html).

To create and set up a `virtualenvwrapper` environment, open a command line window. On Windows, open the Command Prompt.

* Install `virtualenvwrapper`:
    * `pip install virtualenvwrapper`
    * On Windows: `pip install virtualenvwrapper-win`
* Modify system information:
    * On Windows, once you install `virtualenvwrapper`, you may see a warning message saying the install location of `virtualenv` should be added to your PATH. If this happens, add the location listed in the warning message to your PATH. It may be something like this: `C:\Users\UserName\AppData\Local\Packages\...\Python310\Scripts`. Copy the location as listed in your command line window to your user PATH variable. To do this, search *path* in the search bar. Click *Edit the system environment variables*. Under the *Advanced* tab, click *Environment Variables*. Under User variables click *Path*, then *Edit*. Click *New*, then copy-paste the path displayed in the command prompt. To save the changes, click *Ok* on the subsequent 3 windows. Close and re-open the Windows Command Prompt for the changes to take effect.
    * On Linux, follow the instructions for modifying your `.bashrc` file so that the `virtualenv` and `virtualenvwrapper` installation paths are recognized: [Virtualenvwrapper Shell Startup](https://virtualenvwrapper.readthedocs.io/en/latest/install.html#shell-startup-file)
* Now you're ready to create a `virtualenv` environment called `qtuum`:
    * `mkvirtualenv qtuum`
* Navigate to `./notebooks/qtuum`:
    * `cd ./notebooks/qtuum`
* Run the following to install all dependencies inside the environment:
    * `pip install -r requirements.txt`
* Install Jupyter notebook:
    * `pip install notebook`

*Note:* When you run `mkvirtualenv` it automatically activates the new environment for you. You'll see this at the start of the line. In future sessions, the environment will not automatically be activated. If you've already created an environment in a previous session and want to work in it again, run `workon qtuum` to activate it. 

When you are finished running code, deactivate the environment: `deactivate`.

**Option 2: Miniconda**

To create and set up a `conda` environment, open the Anaconda Prompt that was installed when you installed Miniconda. On Windows you can find this in the Start Menu. For more information on `conda` see [conda](https://docs.conda.io/en/latest/)).

* Navigate to `./notebooks/qtuum`:
    * `cd ./notebooks/qtuum`

* Create a `conda` environment from the `yml` file provided, which automatically names the environment to `qtuum`:
    * `conda env create -f environment.yml`

* Activate the environment:
    * `conda activate qtuum`

When you are finished running code, deactivate the environment: `conda deactivate`

### 3. Open Jupyter

You are now ready to run quantum circuits on the Quantinuum hardware!

Navigate to the location of the `notebooks` folder. If you are still in the `qtuum` folder, navigate up one level: `cd ..`

Run `jupyter notebook` and open `1 - Circuit Submissions.ipynb` to get started!

*Note:* Once you have either a `virtualenv` or `conda` environment created, you do not need to go through these steps subsequent times. In the future, open a command line window, activate the environment, and run `jupyter notebook` from the location of your notebooks directory.

---

### Note for Windows Users

When you go to use the code examples and log in using your credentials, you may receive a `RuntimeError` or `SSLError` related to the HTTP or HTTPS connection. This is due to the new Quantinuum URL and the pace at which the Windows certificates are updated. Eventually this will not be needed, but at this time install the `python-certifi-win32` package in your environment.

* Navigate to and activate your environment, then install the following:
    * `pip install python-certifi-win32`