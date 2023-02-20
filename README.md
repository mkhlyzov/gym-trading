# gym-trading
Trading environment for Reinforcement Learning

In order to install run following commands in your home directory:

>git clone https://github.com/mkhlyzov/gym-trading  
>pip install -e ./gym-trading

Currently only editable install is supported due to lack of support for Data Files on developer's end. (Going to be fixed, but not the first priority)

Basic example can be found in [examples](examples/) folder.

Single observation type is a dictionary. One can use gymnasium.wrappers.FlattenObservation wrapper to flatten the observations.