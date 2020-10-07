Usage
=====

Set up a data repository
----------------------------

The following data repository - https://github.com/blackrhinoabm/sabcom-paper is required for SABCom to work.

Open the cloned folder onto your terminal.

Simulation
-----------

To simulate a model using the blow inputs, first make sure that all the files and folders are in your current location.

1. Input folder - example_data (cape_town or johannesburg)
2. Output folder -  example_data/output_file
3. Seed - 2
4. Data output mode - csv-light
5. Scenario - no-intervention. First, make sure that all the files and folders are in your current location.

Next, you type in the inputs in the command line with the below syntax:

``$ simulate -i example_data -o example_data/output_file -s 2 -d csv-light -sc no-intervention``

This will simulate a no_intervention scenario for the seed_2.pkl initialisation. Input files for the city of your choice, and output a csv light data file in the specified output folder (output_file).

Also, note how this assumes that there is already an initialisation file. If there is no initialisation file, then sabcom can be used to produce one given the input files.

Initialisation
---------------
For input folders with no initialisation file. Sabcom uses the "sabcom initialise" function to create an initialisation with the files in input folder.

``$ initialise <input folder path> <seed number>``

``$ sabcom initialise -i example_data -s 3``

As a rule, creating a model initialisation takes much longer than simulating one.

Sample
-------

From the unique spatial structure of the model, we can track how a virus spreads spatially. For example, the figure below shows the quantity of the population infected in different Wards in the City of Cape town.

Note that this concerns a hypothetical simulation of the non-calibrated model and is only used to give an idea of the possible dynamics.

.. image:: /Infected.gif
   :width: 300px
   :height: 500px
   :align: center


Requirements
--------------
Install the required dependencies using requirements.txt.

For **Windows** Users:

``$ python -m pip install -r requirements.txt``

For **Mac** and **Linux** Users

``$ pip install -r requirements.txt``






