Inputs
=======

To simulate a model five arguments are used as inputs:

1.	Path for the input folder (-i) – (cape_town or johannesburg)
2.	Path for the output folder (-o) –(path to output_file)
3.	Seed (-s) – (seed number)
4.	Data output mode (-d) – (csv-light)
5.	Scenario (-sc) – (no-intervention, full lockdown)


Parameters
------------

In the /city/parameters.json file (city being either cape_town/ johannesburg), there are parameters that can be adjusted to have more control of the simulated model.

1. time - Number of days that the model runs. The default number is 365 days.
2. number_of_agents - The total population of people in the model simulation. The default number is 100 000 agents.
3. monte_carlo_runs -
4. exposed_days -
5. asymptom_days -
6. symptom_days -
7. critical_days -

Data
-----
Click on the below to download the sample data files for the following cities:

1. Cape Town
2. Johannesburg










Next, there are two options:-
Simulating the model (using an existing initialisation) or initialising a new model environment that can be used for the simulation.