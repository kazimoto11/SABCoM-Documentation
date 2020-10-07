Modules
========

Agent
------
.. code-block:: python

   class Agent:
       def __init__(self, name, status, district, age_group,
                   informality, number_contacts, district_to_travel_to, compliance):
           """
           This method initialises an agent and its properties.
           Agents have several properties, that can either be state variables (if they change),
           or static parameters. There is a distinction between agent specific parameters
           and parameters that are the same for all agents and could thus be seen
           as global parameters.
           The following inputs are used to initialise the agent
           :param name: unique agent identifier , integer
           :param status: initial disease status, string
           :param coordinates: the geographical coordinates of where the agent lives, tuple (float, float)
           :param district: the unique code / identifier of the district, int
           :param age_group: the age group of the agent, float ([age_0_10', 'age_10_20', 'age_20_30', 'age_30_40', 'age_40_50',
                 'age_50_60', 'age_60_70', 'age_70_80', 'age_80_plus')
           :param informality: a percentage indicating how 'informal' the district the agent lives in is, float (0,1)
           :param number_contacts: the amount of trips the agent will undertake on a daily basis
           """
           # state variables
           self.status = status

           # agent specific parameters
           self.name = name
           self.district = district
           self.age_group = age_group

           # agent specific variables
           self.compliance = compliance
           self.previous_compliance = compliance

           # implementation variables
           self.sick_days = 0
           self.asymptomatic_days = 0
           self.incubation_days = 0
           self.critical_days = 0
           self.exposed_days = 0
           self.days_recovered = 0
           self.others_infected = 0
           self.others_infects_total = 0
           self.travel_neighbours = []
           self.period_to_become_infected = None

           # implementation parameters
           self.num_contacts = number_contacts
           self.household_number = None

           # agent specific parameters that depend on other parameters
           self.district_to_travel_to = district_to_travel_to
           self.informality = informality

       def __repr__(self):
           """
           :return: String representation of the trader
           """
           return self.status + ' Agent' + str(self.name)


Differential equation model
-----------------------------

.. code-block:: python


   import numpy as np

   def differential_equations_model(compartments, t, infection_rate, contact_probability_matrix,
                                   exit_rate_exposed, exit_rate_asymptomatic,
                                   exit_rate_symptomatic, exit_rate_critical,
                                   probability_symptomatic, probability_critical, probability_to_die, hospital_capacity):
       # reshape 63 element vector Z into [7 x 9] matrix
       compartments = compartments.reshape(7, -1)

       # assign rows to disease compartments
       susceptible, exposed, asymptomatic, symptomatic, critical, recovered, dead = compartments

       health_overburdened_multiplier = 1

       # health system can be overburdened which will increase the probability of death
       if critical.sum() > hospital_capacity:
           health_overburdened_multiplier = 1.79 #TODO add this as a parameter
           probability_to_die = np.minimum(health_overburdened_multiplier * probability_to_die, np.ones(9))
           # print(t)

       # construct differential equation evolution equations
       delta_susceptible = -infection_rate * susceptible * contact_probability_matrix.dot((asymptomatic + symptomatic))
       delta_exposed = infection_rate * susceptible * contact_probability_matrix.dot((
               asymptomatic + symptomatic)) - exit_rate_exposed * exposed
       delta_asymptomatic = (1 - probability_symptomatic
                             ) * exit_rate_exposed * exposed - exit_rate_asymptomatic * asymptomatic
       delta_symptomatic = probability_symptomatic * exit_rate_exposed * exposed - exit_rate_symptomatic * symptomatic
       delta_critical = probability_critical * exit_rate_symptomatic * symptomatic - exit_rate_critical * critical
       delta_recovered = exit_rate_asymptomatic * asymptomatic + (
               1 - probability_critical) * exit_rate_symptomatic * symptomatic + (1 - probability_to_die
                                                                                 ) * exit_rate_critical * critical
       delta_dead = probability_to_die * exit_rate_critical * critical

       # store differentials as 63 element vector
       delta_compartments = np.concatenate((delta_susceptible, delta_exposed, delta_asymptomatic,
                                           delta_symptomatic, delta_critical, delta_recovered, delta_dead), axis=0)

       return delta_compartments


Environment
------------

.. code-block:: python

   import numpy as np
   import networkx as nx
   import random
   import copy
   import pandas as pd
   import scipy.stats as stats

   from sabcom.agent import Agent
   from sabcom.helpers import what_informality

   class Environment:
       """
       The environment class contains the agents in a network structure
       """

       def __init__(self, seed, parameters, district_data, age_distribution_per_district,
                   household_contact_matrix, other_contact_matrix, household_size_distribution, travel_matrix):
           """
           This method initialises the environment and its properties.
           :param seed: used to initialise the random generators to ensure reproducibility, int
           :param parameters: contains all model parameters, dictionary
           :param district_data: contains empirical data on the districts, list of tuples
           :param age_distribution_per_district: contains the distribution across age categories per district, dictionary
           :param household_contact_matrix: contains number and age groups for household contacts, Pandas DataFrame
           :param other_contact_matrix: contains number and age groups for all other contacts, Pandas DataFrame
           :param household_size_distribution: contains distribution of household size for all districts, Pandas DataFrame
           :param travel_matrix: contains number and age groups for all other contacts, Pandas DataFrame
           """
           np.random.seed(seed)
           random.seed(seed)
           random.seed(seed)

           self.parameters = parameters
           self.other_contact_matrix = other_contact_matrix

           # 1 create modelled districts
           # 1.1 retrieve population data
           nbd_values = [x[1] for x in district_data]
           population_per_neighbourhood = [x['Population'] for x in nbd_values]

           # 1.2 correct the population in districts to be proportional to number of agents
           correction_factor = sum(population_per_neighbourhood) / parameters["number_of_agents"]
           corrected_populations = [int(round(x / correction_factor)) for x in population_per_neighbourhood]

           # 1.3 only count districts that then have an amount of people bigger than 0
           indices_big_neighbourhoods = [i for i, x in enumerate(corrected_populations) if x > 0]
           corrected_populations_final = [x for i, x in enumerate(corrected_populations) if x > 0]

           # 1.4 create a shock generator for the initialisation of agents initial compliance
           lower, upper = -(parameters['stringency_index'][0] / 100), (1 - (parameters['stringency_index'][0] / 100))
           mu, sigma = 0.0, parameters['private_shock_stdev']
           shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                      size=sum(corrected_populations_final))

           # 1.5 fill up the districts with agents
           self.districts = [x[0] for x in district_data]
           self.district_agents = {d: [] for d in self.districts}
           agents = []
           city_graph = nx.Graph()
           agent_name = 0
           all_travel_districts = {district_data[idx][0]: [] for idx in indices_big_neighbourhoods}

           # for every district
           for num_agents, idx in zip(corrected_populations_final, indices_big_neighbourhoods):
               # 1.5.1 determine district code, informality, and age categories
               district_list = []
               district_code = district_data[idx][0]
               informality = what_informality(district_code, district_data) * parameters["informality_dummy"]

               age_categories = np.random.choice(age_distribution_per_district[district_code].index,
                                                 size=int(num_agents),
                                                 replace=True,
                                                 p=age_distribution_per_district[district_code].values)

               # 1.5.2 determine districts to travel to
               available_districts = list(all_travel_districts.keys())
               probabilities = list(travel_matrix[[str(x) for x in available_districts]].loc[district_code])

               # 1.5.3 add agents to district
               for a in range(num_agents):
                   init_private_signal = parameters['stringency_index'][0] / 100 + shocks[agent_name]
                   district_to_travel_to = np.random.choice(available_districts, size=1, p=probabilities)[0]
                   agent = Agent(agent_name, 's',
                                 district_code,
                                 age_categories[a],
                                 informality,
                                 int(round(other_contact_matrix.loc[age_categories[a]].sum())),
                                 district_to_travel_to,
                                 init_private_signal
                                 )
                   self.district_agents[district_code].append(agent)
                   district_list.append(agent)
                   all_travel_districts[district_to_travel_to].append(agent)
                   agent_name += 1

               # 2 Create the household network structure
               # 2.1 get household size list for this Ward and reduce list to max household size = size of ward
               max_district_household = min(len(district_list), len(household_size_distribution.columns) - 1)
               hh_sizes = household_size_distribution.loc[district_code][:max_district_household]
               # 2.2 then calculate probabilities of this being of a certain size
               hh_probability = pd.Series([float(i) / sum(hh_sizes) for i in hh_sizes])
               hh_probability.index = hh_sizes.index
               # 2.3 determine household sizes
               sizes = []
               while sum(sizes) < len(district_list):
                   sizes.append(int(np.random.choice(hh_probability.index, size=1, p=hh_probability)[0]))
                   hh_probability = hh_probability[:len(district_list) - sum(sizes)]
                   # recalculate probabilities
                   hh_probability = pd.Series([float(i) / sum(hh_probability) for i in hh_probability])
                   try:
                       hh_probability.index = hh_sizes.index[:len(district_list) - sum(sizes)]
                   except:
                       print('Error occured')
                       print('lenght of district list = {}'.format(len(district_list)))
                       print('sum(sizes) = {}'.format(sum(sizes)))
                       print('hh_sizes.index[:len(district_list) - sum(sizes)]is '.format(hh_sizes.index[:len(district_list) - sum(sizes)]))
                       print('hh_probability.index = {}'.format(hh_probability.index))
                       break

               # 2.4 Distribute agents over households
               # 2.4.1 pick the household heads and let it form connections with other based on probabilities.
               # household heads are chosen at random without replacement
               household_heads = np.random.choice(district_list, size=len(sizes), replace=False)
               not_household_heads = [x for x in district_list if x not in household_heads]
               # 2.4.2 let the household heads pick n other agents that are not household heads themselves
               for i, head in enumerate(household_heads):
                   head.household_number = i
                   if sizes[i] > 1:
                       # pick n other agents based on probability given their age
                       p = [household_contact_matrix[to.age_group].loc[head.age_group] for to in not_household_heads]
                       # normalize p
                       p = [float(i) / sum(p) for i in p]
                       household_members = list(np.random.choice(not_household_heads, size=sizes[i]-1, replace=False, p=p))

                       # remove household members from not_household_heads
                       for h in household_members:
                           h.household_number = i
                           not_household_heads.remove(h)

                       # add head to household members:
                       household_members.append(head)
                   else:
                       household_members = [head]

                   # 2.4.3 create graph for household
                   household_graph = nx.Graph()
                   household_graph.add_nodes_from(range(len(household_members)))

                   # create edges between all household members
                   edges = nx.complete_graph(len(household_members)).edges()
                   household_graph.add_edges_from(edges, label='household')

                   # add household members to the agent list
                   agents.append(household_members)

                   # 2.4.4 add network to city graph
                   city_graph = nx.disjoint_union(city_graph, household_graph)

           self.agents = [y for x in agents for y in x]

           # 3 Next, we create the a city wide network structure of recurring contacts based on the travel matrix
           for agent in self.agents:
               agents_to_travel_to = all_travel_districts[agent.district_to_travel_to]
               agents_to_travel_to.remove(agent)  # remove the agent itself

               if agents_to_travel_to:
                   # select the agents which it is most likely to have contact with based on the travel matrix
                   p = [other_contact_matrix[a.age_group].loc[agent.age_group] for a in agents_to_travel_to]
                   # normalize p
                   p = [float(i) / sum(p) for i in p]

                   location_closest_agents = np.random.choice(agents_to_travel_to,
                                                             size=min(agent.num_contacts, len(agents_to_travel_to)),
                                                             replace=False,
                                                             p=p)

                   for ca in location_closest_agents:
                       city_graph.add_edge(agent.name, ca.name, label='other')

           self.network = city_graph

           # 4 rename agents to reflect their new position
           for idx, agent in enumerate(self.agents):
               agent.name = idx

           # 5 add agent to the network structure
           for idx, agent in enumerate(self.agents):
               self.network.nodes[idx]['agent'] = agent

           self.infection_states = []
           self.infection_quantities = {key: [] for key in ['e', 's', 'i1', 'i2', 'c', 'r', 'd', 'compliance']}

           # 6 add stringency index from parameters to reflect how strict regulations are enforced
           self.stringency_index = parameters['stringency_index']
           if len(parameters['stringency_index']) < parameters['time']:
               self.stringency_index += [parameters['stringency_index'][-1] for x in range(len(
                   parameters['stringency_index']), parameters['time'])]

       def store_network(self):
           """Returns a deep copy of the current network"""
           current_network = copy.deepcopy(self.network)
           return current_network

       def write_status_location(self, period, seed, base_folder='measurement/'):
           """
           Writes information about the agents and their status in the current period to a csv file
           :param period: the current time period, int
           :param seed: used to initialise the random generators to ensure reproducibility, int
           :param base_folder: the location of the folder to write the csv to, string
           :return: None
           """
           location_status_data = {'agent': [], 'status': [], 'WardID': [], 'age_group': [],
                                   'others_infected': [], 'compliance': []}
           for agent in self.agents:
               location_status_data['agent'].append(agent.name)
               location_status_data['status'].append(agent.status)
               location_status_data['WardID'].append(agent.district)
               location_status_data['age_group'].append(agent.age_group)
               location_status_data['others_infected'].append(agent.others_infected)
               location_status_data['compliance'].append(agent.compliance)

           pd.DataFrame(location_status_data).to_csv(base_folder + "seed" + str(seed) + "/agent_data{0:04}.csv".format(
               period))

           # output links
           if period == 0:
               pd.DataFrame(self.network.edges()).to_csv(base_folder + "seed" + str(seed) + "/edge_list{0:04}.csv".format(
                   period))



Estimation
-----------

.. code-block:: python


   import numpy as np
   import pandas as pd
   import json
   import math
   import os
   import scipy.stats as stats
   import scipy.optimize as sciopt

   from sabcom.updater import updater

    def ls_model_performance(input_params, input_folder_path, mc_runs, output_folder_path, scenario, names):
       """
       Simple function calibrate uncertain model parameters
       :param input_parameters: list of input parameters
       :return: cost
       """
       # zip names and input params together in a dictionary
       #new_params = {name: par for name, par in zip(names, input_params)}

       # transmission_probability = input_params[0]
       # initial_infections = input_params[1]
       # infection_multiplier = input_params[2]
       # base_awareness_likelihood = input_params[3]
       # gathering_max_contacts = input_params[4]

       parameter_json_path = os.path.join(input_folder_path, 'parameters.json')
       mc_runs = mc_runs

       # open estimation_parameter.json file to extract special parameters used for the estimation
       with open(parameter_json_path) as json_file:
           param_file = json.load(json_file)

       emp_fatality_curve = param_file['empirical_fatalities']
       empirical_population = param_file['empirical_population']
       # new!
       # param_file['private_shock_stdev'] = input_params[5]
       # param_file['weight_private_signal'] = input_params[6]
       # # change all of them
       # param_file['probability_transmission'] = transmission_probability
       # param_file["physical_distancing_multiplier"] = infection_multiplier
       # param_file['likelihood_awareness'] = base_awareness_likelihood
       # param_file["gathering_max_contacts"] = gathering_max_contacts
       for name, par in zip(names, input_params):
           if name not in ['visiting_recurring_contacts_multiplier', 'total_initial_infections']:
               param_file[name] = par
           if name == 'visiting_recurring_contacts_multiplier':
               vis_rec = par
           else:
               vis_rec = None
           if name == 'total_initial_infections':
               initial_infections = par
           else:
               vis_rec = None

       param_file["time"] = len(emp_fatality_curve)

       # dump in config file
       with open('estimation_parameters.json', 'w') as outfile:
           json.dump(param_file, outfile)

       costs = []
       for seed in range(mc_runs):
           # check if the seed can be found in the input folder, if not skip seed
           inititialisation_path = os.path.join(input_folder_path, 'initialisations')
           seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))

           if not os.path.exists('{}'.format(seed_path)):
               print('Path {} not found, will continue loop'.format(seed_path))
               continue

           # run model with parameters.
           environment = updater(input_folder_path=input_folder_path,
                                 output_folder_path=output_folder_path, seed=seed,
                                 scenario=scenario,
                                 initial_infections=initial_infections,
                                 visiting_recurring_contacts_multiplier=vis_rec,
                                 stringency_changed=True,
                                 sensitivity_config_file_path='estimation_parameters.json')

           sim_dead_curve = pd.DataFrame(environment.infection_quantities)['d'] * (empirical_population / param_file['number_of_agents'])
           sim_dead_curve = sim_dead_curve.diff().ewm(span=10).mean()

           # calculate the cost
           costs.append(ls_cost_function(emp_fatality_curve, sim_dead_curve))

       return np.mean(costs)

     def confidence_interval(data, av):
     sample_stdev = np.std(data)
     sigma = sample_stdev/math.sqrt(len(data))
     return stats.t.interval(alpha = 0.95, df= 24, loc=av, scale=sigma)

    def ls_cost_function(observed_values, average_simulated_values):
       """
       Simple cost function to calculate average squared deviation of simulated values from observed values
       :param observed_values: list of observed data points
       :param average_simulated_values: list of corresponding observed data points
       :return:
       """
       score = 0
       for x, y in zip(observed_values, average_simulated_values):
           if x > 0.0:
               score += np.true_divide((x - y), x)**2
           else:
               score += 0

       if np.isnan(score):
           return np.inf
       else:
           return score

   def cost_function(observed_values, average_simulated_values):
       """
       Simple cost function to calculate average squared deviation of simulated values from observed values
       :param observed_values: dictionary of observed stylized facts
       :param average_simulated_values: dictionary of corresponding simulated stylized facts
       :return:
       """
       score = 0
       for key in observed_values:
           score += np.true_divide((observed_values[key] - average_simulated_values[key]), observed_values[key])**2

       if np.isnan(score):
           return np.inf
       else:
           return score


   # =====================================================================================================================================
   # Copyright
   # =====================================================================================================================================

   # Copyright (C) 2017 Alexander Blaessle.
   # This software is distributed under the terms of the GNU General Public License.

   # This file is part of constNMPY.

   # constNMPy is a small python package allowing to run a Nelder-Mead optimization via scipy's fmin function.

   # You should have received a copy of the GNU General Public License
   # along with this program.  If not, see <http://www.gnu.org/licenses/>.


   # ===========================================================================================================================================================================
   # Module Functions
   # ===========================================================================================================================================================================

   def constrNM(func, x0, LB, UB, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=0,
               retall=0, callback=None):
       """Constrained Nelder-Mead optimizer.
       Transforms a constrained problem
       Args:
           func (function): Objective function.
           x0 (numpy.ndarray): Initial guess.
           LB (numpy.ndarray): Lower bounds.
           UB (numpy.ndarray): Upper bounds.
       Keyword Args:
           args (tuple): Extra arguments passed to func, i.e. ``func(x,*args).``
           xtol (float) :Absolute error in xopt between iterations that is acceptable for convergence.
           ftol(float) : Absolute error in ``func(xopt)`` between iterations that is acceptable for convergence.
           maxiter(int) : Maximum number of iterations to perform.
           maxfun(int) : Maximum number of function evaluations to make.
           full_output(bool) : Set to True if fopt and warnflag outputs are desired.
           disp(bool) : Set to True to print convergence messages.
           retall(bool): Set to True to return list of solutions at each iteration.
           callback(callable) : Called after each iteration, as ``callback(xk)``, where xk is the current parameter vector.
       """

       # Check input
       if len(LB) != len(UB) or len(LB) != len(x0):
           raise ValueError('Input arrays have unequal size.')

       # Check if x0 is within bounds
       for i, x in enumerate(x0):

           if LB[i] is not None:
               if x < LB[i]:
                   errStr = 'Initial guess x0[' + str(i) + ']=' + str(x) + ' out of bounds.'
                   raise ValueError(errStr)

           if UB[i] is not None:
               if x > UB[i]:
                   errStr = 'Initial guess x0[' + str(i) + ']=' + str(x) + ' out of bounds.'
                   raise ValueError(errStr)

       # Transform x0
       x0 = transformX0(x0, LB, UB)

       # Stick everything into args tuple
       opts = tuple([func, LB, UB, args])

       # Call fmin
       res = sciopt.fmin(constrObjFunc, x0, args=opts, ftol=ftol, xtol=xtol, maxiter=maxiter, disp=disp,
                         full_output=full_output, callback=callback, maxfun=maxfun, retall=retall)

       # Convert res to list
       res = list(res)

       # Dictionary for results
       rDict = {'fopt': None, 'iter': None, 'funcalls': None, 'warnflag': None, 'xopt': None, 'allvecs': None}

       # Transform back results
       if full_output or retall:
           r = transformX(res[0], LB, UB)
       else:
           r = transformX(res, LB, UB)
       rDict['xopt'] = r

       # If full_output is selected, enter all results in dict
       if full_output:
           rDict['fopt'] = res[1]
           rDict['iter'] = res[2]
           rDict['funcalls'] = res[3]
           rDict['warnflag'] = res[4]

       # If retall is selected, transform back all values and append to dict
       if retall:
           allvecs = []
           for r in res[-1]:
               allvecs.append(transformX(r, LB, UB))
           rDict['allvecs'] = allvecs

       return rDict

   def constrObjFunc(x, func, LB, UB, args):
       r"""Objective function when using Constrained Nelder-Mead.
       Calls :py:func:`TransformX` to transform ``x`` into
       constrained version, then calls objective function ``func``.
       Args:
           x (numpy.ndarray): Input vector.
           func (function): Objective function.
           LB (numpy.ndarray): Lower bounds.
           UB (numpy.ndarray): Upper bounds.
       Keyword Args:
           args (tuple): Extra arguments passed to func, i.e. ``func(x,*args).``
       Returns:
           float: Return value of ``func(x,*args)``.
       """

       # print x
       x = transformX(x, LB, UB)
       # print x
       # raw_input()

       return func(x, *args)


   def transformX(x, LB, UB, offset=1E-20):
       r"""Transforms ``x`` into constrained form, obeying upper bounds ``UB`` and lower bounds ``LB``.
       .. note:: Will add tiny offset to LB if ``LB[i]=0``, to avoid singularities.
       Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
       Args:
           x (numpy.ndarray): Input vector.
           LB (numpy.ndarray): Lower bounds.
           UB (numpy.ndarray): Upper bounds.
       Keyword Args:
           offset (float): Small offset added to lower bound if LB=0.
       Returns:
           numpy.ndarray: Transformed x-values.
       """

       # Make sure everything is float
       x = np.asarray(x, dtype=np.float64)
       # LB=np.asarray(LB,dtype=np.float64)
       # UB=np.asarray(UB,dtype=np.float64)

       # Add offset if necessary to avoid singularities
       for l in LB:
           if l == 0:
               l = l + offset

       # Determine number of parameters to be fitted
       nparams = len(x)

       # Make empty vector
       xtrans = np.zeros(np.shape(x))

       # k allows some variables to be fixed, thus dropped from the
       # optimization.
       k = 0

       for i in range(nparams):

           # Upper bound only
           if UB[i] != None and LB[i] == None:

               xtrans[i] = UB[i] - x[k] ** 2
               k = k + 1

           # Lower bound only
           elif UB[i] == None and LB[i] != None:

               xtrans[i] = LB[i] + x[k] ** 2
               k = k + 1

           # Both bounds
           elif UB[i] != None and LB[i] != None:

               xtrans[i] = (np.sin(x[k]) + 1.) / 2. * (UB[i] - LB[i]) + LB[i]
               xtrans[i] = max([LB[i], min([UB[i], xtrans[i]])])
               k = k + 1

           # No bounds
           elif UB[i] == None and LB[i] == None:

               xtrans[i] = x[k]
               k = k + 1

           # NOTE: The original file has here another case for fixed variable. We might need to add this here!!!

       return np.array(xtrans)


   def transformX0(x0, LB, UB):
       r"""Transforms ``x0`` into constrained form, obeying upper bounds ``UB`` and lower bounds ``LB``.
       Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
       Args:
           x0 (numpy.ndarray): Input vector.
           LB (numpy.ndarray): Lower bounds.
           UB (numpy.ndarray): Upper bounds.
       Returns:
           numpy.ndarray: Transformed x-values.
       """

       # Turn into list
       x0u = list(x0)

       k = 0
       for i in range(len(x0)):

           # Upper bound only
           if UB[i] != None and LB[i] == None:
               if UB[i] <= x0[i]:
                   x0u[k] = 0
               else:
                   x0u[k] = np.sqrt(UB[i] - x0[i])
               k = k + 1

           # Lower bound only
           elif UB[i] == None and LB[i] != None:
               if LB[i] >= x0[i]:
                   x0u[k] = 0
               else:
                   x0u[k] = np.sqrt(x0[i] - LB[i])
               k = k + 1


           # Both bounds
           elif UB[i] != None and LB[i] != None:
               if UB[i] <= x0[i]:
                   x0u[k] = np.pi / 2
               elif LB[i] >= x0[i]:
                   x0u[k] = -np.pi / 2
               else:
                   x0u[k] = 2 * (x0[i] - LB[i]) / (UB[i] - LB[i]) - 1;
                   # shift by 2*pi to avoid problems at zero in fmin otherwise, the initial simplex is vanishingly small
                   x0u[k] = 2 * np.pi + np.arcsin(max([-1, min(1, x0u[k])]));
               k = k + 1

           # No bounds
           elif UB[i] == None and LB[i] == None:
               x0u[k] = x0[i]
               k = k + 1

       return np.array(x0u)


   def printAttr(name, attr, maxL=5):
       """Prints single attribute in the form attributeName = attributeValue.
       If attributes are of type ``list`` or ``numpy.ndarray``, will check if the size
       exceeds threshold. If so, will only print type and dimension of attribute.
       Args:
           name (str): Name of attribute.
           attr (any): Attribute value.
       Keyword Args:
           maxL (int): Maximum length threshold.
       """

       if isinstance(attr, (list)):
           if len(attr) > maxL:
               print(name, " = ", getListDetailsString(attr))
               return True
       elif isinstance(attr, (np.ndarray)):
           if min(attr.shape) > maxL:
               print(name, " = ", getArrayDetailsString(attr))
               return True

       print(name, " = ", attr)

       return True


   def getListDetailsString(l):
       """Returns string saying "List of length x", where x is the length of the list.
       Args:
           l (list): Some list.
       Returns:
           str: Printout of type and length.
       """

       return "List of length " + str(len(l))


   def getArrayDetailsString(l):
       """Returns string saying "Array of shape x", where x is the shape of the array.
       Args:
           l (numpy.ndarray): Some array.
       Returns:
           str: Printout of type and shape.
       """

       return "Array of shape " + str(l.shape)


   def printDict(dic, maxL=5):
       """Prints all dictionary entries in the form key = value.
       If attributes are of type ``list`` or ``numpy.ndarray``, will check if the size
       exceeds threshold. If so, will only print type and dimension of attribute.
       Args:
           dic (dict): Dictionary to be printed.
       Returns:
           bool: True
       """

       for k in dic.keys():
           printAttr(k, dic[k], maxL=maxL)

       return True


Helpers
---------

.. code-block:: python


   import random
   import numpy as np
   import pandas as pd
   import math
   from sklearn import preprocessing
   import scipy.stats as stats


   def edge_in_cliq(edge, nodes_in_cliq):
       if edge[0] in nodes_in_cliq:
           return True
       else:
           return False


   def edges_to_remove_neighbourhood(all_edges, neighbourhood_density, nbh_nodes):
       neighbourhood_edges = [e for e in all_edges if edge_in_cliq(e, nbh_nodes)]
       sample_size = int(len(neighbourhood_edges) * (1-neighbourhood_density))
       # sample random edges
       chosen_edges = random.sample(neighbourhood_edges, sample_size)
       return chosen_edges


   def what_neighbourhood(index, neighbourhood_nodes):
       for n in neighbourhood_nodes:
           if index in neighbourhood_nodes[n]:
               return n

       raise ValueError('Neighbourhood not found.')


   def what_coordinates(neighbourhood_name, dataset):
      for x in range(len(dataset)):
           if neighbourhood_name in dataset[x]:
               return dataset[x][1]['lon'], dataset[x][1]['lat'],

       raise ValueError("Corresponding coordinates not found")


   def what_informality(neighbourhood_name, dataset):
       for x in range(len(dataset)):
           if neighbourhood_name in dataset[x]:
               try:
                   return dataset[x][1]['Informal_residential']
               except:
                   return None

       raise ValueError("Corresponding informality not found")


   def confidence_interval(data, av):
       sample_stdev = np.std(data)
       sigma = sample_stdev/math.sqrt(len(data))
       return stats.t.interval(alpha=0.95, df=24, loc=av, scale=sigma)


   def generate_district_data(number_of_agents, path, max_districts=None):
       """
       Transforms input data on informal residential, initial infections, and population and transforms it to
       a list of organised data for the simulation.
       :param number_of_agents: number of agents in the simulation, integer
       :param max_districts: (optional) maximum amount of districts simulated, integer
       :return: data set containing district data for simulation, list
       """
       informal_residential = pd.read_csv('{}/f_informality.csv'.format(path))#.iloc[:-1]
       inital_infections = pd.read_csv('{}/f_initial_cases.csv'.format(path), index_col=1)
       inital_infections = inital_infections.sort_index()
       population = pd.read_csv('{}/f_population.csv'.format(path))

       # normalise district informality
       x = informal_residential[['Informal_residential']].values.astype(float)
       min_max_scaler = preprocessing.MinMaxScaler()
       x_scaled = min_max_scaler.fit_transform(x)
       informal_residential['Informal_residential'] = pd.DataFrame(x_scaled)
       population['Informal_residential'] = informal_residential['Informal_residential']

       # determine smallest district based on number of agents
       smallest_size = population['Population'].sum() / number_of_agents

       # generate data set for model input
       districts_data = []
       for i in range(len(population)):
           if population['Population'].iloc[i] > smallest_size:
               districts_data.append(
                   [int(population['WardID'].iloc[i]), {'Population': population['Population'].iloc[i],
                                                       #'lon': population['lon'].iloc[i],
                                                       #'lat': population['lat'].iloc[i],
                                                       'Informal_residential': population['Informal_residential'].iloc[i],
                                                       'Cases_With_Subdistricts':
                                                           inital_infections.loc[population['WardID'].iloc[i]][
                                                               'Cases'],
                                                       },
                   ])

       if max_districts is None:
           max_districts = len(districts_data)  # this can be manually shortened to study dynamics in some districts

       return districts_data[:max_districts]

Runner
-------

.. code-block:: python


   import random
   import numpy as np
   import scipy.stats as stats


   def runner(environment, initial_infections, seed, data_folder='output_data/',
             data_output=False, calculate_r_naught=False):
       """
       This function is used to run / simulate the model.
       :param environment: contains the parameters and agents, Environment object
       :param initial_infections: contains the Wards and corresponding initial infections, Pandas DataFrame
       :param seed: used to initialise the random generators to ensure reproducibility, int
       :param data_folder:  string of the folder where data output files should be created
       :param data_output:  can be 'csv', 'network', or False (for no output)
       :param calculate_r_naught: set to True to calculate the R0 that the model produces given a single infected agent
       :return: environment object containing the updated agents, Environment object
       """
       # 1 set monte carlo seed
       np.random.seed(seed)
       random.seed(seed)

       # 2 create sets for all agent types
       dead = []
       recovered = []
       critical = []
       sick_with_symptoms = []
       sick_without_symptoms = []
       exposed = []
       susceptible = [agent for agent in environment.agents]
       compliance = []

       # 3 Initialisation of infections
       # 3a infect a fixed initial agent to calculate R_0
       if calculate_r_naught:
           initial_infected = []
           chosen_agent = environment.agents[environment.parameters['init_infected_agent']]
           chosen_agent.status = 'e'
           initial_infected.append(chosen_agent)
           exposed.append(chosen_agent)
           susceptible.remove(chosen_agent)
       # 3b the default mode is to infect a set of agents based on the locations of observed infections
       else:
           initial_infections = initial_infections.sort_index()
           cases = [x for x in initial_infections['Cases']]
           probabilities_new_infection_district = [float(i) / sum(cases) for i in cases]

           initial_infected = []
           # 3b-1 select districts with probability
           chosen_districts = list(np.random.choice(environment.districts,
                                                   environment.parameters['total_initial_infections'],
                                                   p=probabilities_new_infection_district))
           # 3b-2 count how often a district is in that list
           chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                         chosen_districts.count(distr)) for distr in chosen_districts}

           for district in chosen_districts:
               # 3b-3 infect appropriate number of random agents
               chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                               replace=False)
               categories = ['e', 'i1', 'i2']
               # 3b-4 and give them a random status exposed, asymptomatic, or symptomatic with a random number of days
               # already passed being in that state
               for chosen_agent in chosen_agents:
                   new_status = random.choice(categories)
                   chosen_agent.status = new_status
                   if new_status == 'e':
                       chosen_agent.incubation_days = np.random.randint(0, environment.parameters['exposed_days'])
                       exposed.append(chosen_agent)
                   elif new_status == 'i1':
                       chosen_agent.asymptomatic_days = np.random.randint(0, environment.parameters['asymptom_days'])
                       sick_without_symptoms.append(chosen_agent)
                   elif new_status == 'i2':
                       chosen_agent.sick_days = np.random.randint(0, environment.parameters['symptom_days'])
                       sick_with_symptoms.append(chosen_agent)

                   susceptible.remove(chosen_agent)

       # 4 day loop
       for t in range(environment.parameters["time"]):
           # 4.1 check if the health system is not overburdened
           if len(critical) / len(environment.agents) > environment.parameters["health_system_capacity"]:
               health_overburdened_multiplier = environment.parameters["no_hospital_multiplier"]
           else:
               health_overburdened_multiplier = 1.0

           # 4.2 create a generator to generate shocks for private signal for this period based on current stringency index
           lower, upper = -(environment.stringency_index[t] / 100), (1 - (environment.stringency_index[t] / 100))
           mu, sigma = 0.0, environment.parameters['private_shock_stdev']
           shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                       size=len(susceptible + exposed + sick_without_symptoms + sick_with_symptoms + critical + recovered))

           # 4.3 update status loop for all agents, except dead agents
           for i, agent in enumerate(susceptible + exposed + sick_without_symptoms + sick_with_symptoms + critical + recovered):
               # 4.3.1 save compliance to previous compliance
               agent.previous_compliance = agent.compliance

               # 4.3.2 calculate new compliance based on private and neighbour signal
               neighbours_to_learn_from = [environment.agents[x] for x in environment.network.neighbors(agent.name)]

               private_signal = environment.stringency_index[t] / 100 + shocks[i]
               if neighbours_to_learn_from:  # take into account the scenario that there are no neighbours to learn from
                   neighbour_signal = np.mean([x.previous_compliance for x in neighbours_to_learn_from])
               else:
                   neighbour_signal = private_signal

               agent.compliance = (1 - agent.informality) * \
                                 (environment.parameters['weight_private_signal'] * private_signal +
                                   (1 - environment.parameters['weight_private_signal']) * neighbour_signal)

               # 4.3.3 the disease status of the agent
               if agent.status == 's' and agent.period_to_become_infected == t:
                   agent.status = 'e'
                   susceptible.remove(agent)
                   exposed.append(agent)

               elif agent.status == 'e':
                   agent.exposed_days += 1
                   # some agents will become infectious but do not show agents while others will show symptoms
                   if agent.exposed_days > environment.parameters["exposed_days"]:
                       if np.random.random() < environment.parameters["probability_symptomatic"]:
                           agent.status = 'i2'
                           exposed.remove(agent)
                           sick_with_symptoms.append(agent)
                       else:
                           agent.status = 'i1'
                           exposed.remove(agent)
                           sick_without_symptoms.append(agent)

               # any agent with status i1, or i2 might first infect other agents and then will update her status
               elif agent.status in ['i1', 'i2']:
                   # check if the agent is aware that is is infected here and set compliance to 1.0 if so
                   likelihood_awareness = environment.parameters['likelihood_awareness']
                   if np.random.random() < likelihood_awareness and agent.status == 'i2':
                       agent.compliance = 1.0

                   # Infect others / TAG SUSCEPTIBLE AGENTS FOR INFECTION
                   agent.others_infected = 0

                   # find indices from neighbour agents
                   household_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                           environment.agents[x].household_number == agent.household_number and
                                           environment.agents[x].district == agent.district]
                   other_neighbours = [x for x in environment.network.neighbors(agent.name) if
                                       environment.agents[x].household_number != agent.household_number or
                                       environment.agents[x].district != agent.district]

                   # depending on compliance, the amount of non-household contacts an agent can visit is reduced
                   visiting_r_contacts_multiplier = environment.parameters["visiting_recurring_contacts_multiplier"][t]
                   compliance_term_contacts = (1 - visiting_r_contacts_multiplier) * (1 - agent.compliance)

                   # step 1 planned contacts is shaped by
                   if other_neighbours:
                       planned_contacts = int(round(len(other_neighbours
                                                       ) * (visiting_r_contacts_multiplier + compliance_term_contacts)))
                   else:
                       planned_contacts = 0

                   # step 2 by gathering max contacts
                   gathering_max_contacts = environment.parameters['gathering_max_contacts']
                   if gathering_max_contacts != float('inf'):
                       gathering_max_contacts = round(
                          gathering_max_contacts * (1 + (1 - (environment.stringency_index[t] / 100))))
                       individual_max_contacts = int(round(gathering_max_contacts * (1 + (1 - agent.compliance))))
                   else:
                       individual_max_contacts = gathering_max_contacts

                   if planned_contacts > individual_max_contacts:
                       other_neighbours = random.sample(other_neighbours, individual_max_contacts)
                   else:
                       other_neighbours = random.sample(other_neighbours, planned_contacts)

                   # step 3 combine household neighbours with other neighbours
                   neighbours_from_graph = household_neighbours + other_neighbours
                   # step 4 find the corresponding agents and add them to a list to infect
                   if agent.compliance == 1.0:
                       neighbours_to_infect = [environment.agents[idx] for idx in household_neighbours]
                   else:
                       neighbours_to_infect = [environment.agents[idx] for idx in neighbours_from_graph]
                   # step 4 let these agents be infected (with random probability
                   physical_distancing_multiplier = environment.parameters["physical_distancing_multiplier"]
                   for neighbour in neighbours_to_infect:
                       if neighbour.household_number == agent.household_number and neighbour.district == agent.district:
                           compliance_term_phys_dis = 0.0  # (1 - physical_distancing_multiplier)
                           compliance_term_phys_dis_neighbour = 0.0
                       else:
                           compliance_term_phys_dis = (1 - physical_distancing_multiplier) * (1 - agent.compliance)
                           compliance_term_phys_dis_neighbour = (1 - physical_distancing_multiplier) * (
                                       1 - neighbour.compliance)

                       if neighbour.status == 's' and np.random.random() < (
                               environment.parameters['probability_transmission'] * (
                               physical_distancing_multiplier + compliance_term_phys_dis) * (
                                       physical_distancing_multiplier + compliance_term_phys_dis_neighbour)):
                           neighbour.period_to_become_infected = t + 1
                           agent.others_infected += 1
                           agent.others_infects_total += 1

                   # update current status based on category
                   if agent.status == 'i1':
                       agent.asymptomatic_days += 1
                       # asymptomatic agents all recover after some time
                       if agent.asymptomatic_days > environment.parameters["asymptom_days"]:
                           # calculate R0 here if the first agent recovers
                           if calculate_r_naught and agent in initial_infected:
                               print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                               return agent.others_infects_total

                           agent.status = 'r'
                           sick_without_symptoms.remove(agent)
                           recovered.append(agent)

                   elif agent.status == 'i2':
                       agent.sick_days += 1
                       # some symptomatic agents recover
                       if agent.sick_days > environment.parameters["symptom_days"]:
                           if np.random.random() < environment.parameters["probability_critical"][agent.age_group]:
                               agent.status = 'c'
                               sick_with_symptoms.remove(agent)
                               critical.append(agent)
                           else:
                               # calculate R0 here if the first agent recovers
                               if calculate_r_naught and agent in initial_infected:
                                   print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                                   return agent.others_infects_total
                               agent.status = 'r'
                               sick_with_symptoms.remove(agent)
                               recovered.append(agent)

               elif agent.status == 'c':
                   agent.compliance = 1.0
                   agent.critical_days += 1
                   # some agents in critical status will die, the rest will recover
                   if agent.critical_days > environment.parameters["critical_days"]:
                       # calculate R0 here if the first agent recovers or dies
                       if calculate_r_naught and agent in initial_infected:
                           print(t, ' patient zero recovered or dead with R0 = ', agent.others_infects_total)
                           return agent.others_infects_total

                       if np.random.random() < (environment.parameters["probability_to_die"][
                                   agent.age_group] * health_overburdened_multiplier):
                           agent.status = 'd'
                           critical.remove(agent)
                           dead.append(agent)
                       else:
                           agent.status = 'r'
                           critical.remove(agent)
                           recovered.append(agent)

               elif agent.status == 'r':
                   agent.days_recovered += 1
                   if np.random.random() < (environment.parameters["probability_susceptible"] * agent.days_recovered):
                       recovered.remove(agent)
                       agent.status = 's'
                       susceptible.append(agent)

               compliance.append(agent.compliance)

           # New infections
           if t == environment.parameters['time_4_new_infections']:
               if environment.parameters['new_infections_scenario'] == 'initial':
                   cases = [x for x in initial_infections['Cases']]
                   probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                   # select districts with probability
                   chosen_districts = list(np.random.choice(environment.districts,
                                                           environment.parameters['second_infection_n'],
                                                           p=probabilities_second_infection_district))
                   # count how often a district is in that list
                   chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                                 chosen_districts.count(distr)) for distr in chosen_districts}

               elif environment.parameters['new_infections_scenario'] == 'random':
                   cases = [1 for x in initial_infections['Cases']]
                   probabilities_second_infection_district = [float(i) / sum(cases) for i in cases]
                   # select districts with probability
                   chosen_districts = list(np.random.choice(environment.districts,
                                                           environment.parameters['second_infection_n'],
                                                           p=probabilities_second_infection_district))
                   # count how often a district is in that list
                   chosen_districts = {distr: min(len(environment.district_agents[distr]),
                                                 chosen_districts.count(distr)) for distr in chosen_districts}
               else:
                   chosen_districts = []

               for district in chosen_districts:
                   # infect appropriate number of random agents
                   chosen_agents = np.random.choice(environment.district_agents[district], chosen_districts[district],
                                                   replace=False)
                   categories = ['e', 'i1', 'i2']
                   for chosen_agent in chosen_agents:
                       if chosen_agent.status == 's':
                            new_status = random.choice(categories)
                           chosen_agent.status = new_status
                           if new_status == 'e':
                               chosen_agent.incubation_days = np.random.randint(0, environment.parameters['exposed_days'])
                               exposed.append(chosen_agent)
                           elif new_status == 'i1':
                               chosen_agent.asymptomatic_days = np.random.randint(0,
                                                                                 environment.parameters['asymptom_days'])
                               sick_without_symptoms.append(chosen_agent)
                           elif new_status == 'i2':
                               chosen_agent.sick_days = np.random.randint(0, environment.parameters['symptom_days'])
                               sick_with_symptoms.append(chosen_agent)

                           susceptible.remove(chosen_agent)

           if data_output == 'network':
               environment.infection_states.append(environment.store_network())
           elif data_output == 'csv':
               environment.write_status_location(t, seed, data_folder)
           elif data_output == 'csv-light':
               # save only the total quantity of agents per category
               for key, quantity in zip(['e', 's', 'i1', 'i2',
                                         'c', 'r', 'd'],
                                       [exposed, susceptible, sick_without_symptoms, sick_with_symptoms,
                                         critical, recovered, dead]):
                   environment.infection_quantities[key].append(len(quantity))
               environment.infection_quantities['compliance'].append(np.mean(compliance))

       return environment

Updater
-------

.. code-block:: python


   import click
   import os
   import pickle
   import json
   import logging
   import pandas as pd
   import scipy.stats as stats

   from sabcom.runner import runner
   from sabcom.helpers import generate_district_data, what_informality


   def updater(**kwargs):
       """
       :param kwargs:
       :return:
       """

       # store often used arguments in temporary variable
       seed = kwargs.get('seed')
       output_folder_path = kwargs.get('output_folder_path')
       input_folder_path = kwargs.get('input_folder_path')

       # formulate paths to initialisation folder and seed within input folder
       inititialisation_path = os.path.join(input_folder_path, 'initialisations')
       seed_path = os.path.join(inititialisation_path, 'seed_{}.pkl'.format(seed))
       if not os.path.exists(seed_path):
           click.echo(seed_path + ' not found', err=True)
           click.echo('Error: specify a valid seed')
           return

       # open the seed pickle object as an environment
       data = open(seed_path, "rb")
       list_of_objects = pickle.load(data)
       environment = list_of_objects[0]

       # initialise logging
       logging.basicConfig(filename=os.path.join(output_folder_path,
                                                 'simulation_seed{}.log'.format(seed)), filemode='w', level=logging.DEBUG)
       logging.info('Start of simulation seed{} with arguments -i ={}, -o={}'.format(seed,
                                                                                     input_folder_path,
                                                                                     output_folder_path))

       # update optional parameters
       if kwargs.get('sensitivity_config_file_path'):
           config_path = kwargs.get('sensitivity_config_file_path')
           if not os.path.exists(config_path):
               click.echo(config_path + ' not found', err=True)
               click.echo('Error: specify a valid path to the sensitivity config file')
               return
           else:
               with open(config_path) as json_file:
                   config_file = json.load(json_file)

                   for param in config_file:
                       environment.parameters[param] = config_file[param]

       if kwargs.get('days'):
           environment.parameters['time'] = kwargs.get('days')
           click.echo('Time has been set to {}'.format(environment.parameters['time']))
           logging.debug('Time has been set to {}'.format(environment.parameters['time']))
           # ensure that stringency is never shorter than time if time length is increased
           if len(environment.stringency_index) < environment.parameters['time']:
               environment.stringency_index += [environment.stringency_index[-1] for x in range(
                   len(environment.stringency_index), environment.parameters['time'])]
           logging.debug('The stringency index has been lenghtened by {}'.format(
               environment.parameters['time'] - len(environment.stringency_index)))

       if kwargs.get('probability_transmission'):
           environment.parameters['probability_transmission'] = kwargs.get('probability_transmission')
           click.echo(
               'Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))
           logging.debug(
               'Transmission probability has been set to {}'.format(environment.parameters['probability_transmission']))

       if kwargs.get('second_infection_n'):
           environment.parameters['second_infection_n'] = kwargs.get('second_infection_n')
           click.echo(
               'Second infection number has been set to {}'.format(environment.parameters['second_infection_n']))
           logging.debug(
               'Second infection number has been set to {}'.format(environment.parameters['second_infection_n']))

       if kwargs.get('time_4_new_infections'):
           environment.parameters['time_4_new_infections'] = kwargs.get('time_4_new_infections')
           click.echo(
               'Second infection time has been set to {}'.format(environment.parameters['time_4_new_infections']))
           logging.debug(
               'Second infection time has been set to {}'.format(environment.parameters['time_4_new_infections']))

       if kwargs.get('new_infections_scenario'):
           environment.parameters['new_infections_scenario'] = kwargs.get('new_infections_scenario')
           click.echo(
               'New infections scenario has been set to {}'.format(environment.parameters['new_infections_scenario']))
           logging.debug(
               'New infections scenario has been set to {}'.format(environment.parameters['new_infections_scenario']))

       if kwargs.get('visiting_recurring_contacts_multiplier'):
           environment.parameters['visiting_recurring_contacts_multiplier'] = [
               kwargs.get('visiting_recurring_contacts_multiplier') for x in range(environment.parameters['time'])]
           click.echo('Recurring contacts has been set to {}'.format(
               environment.parameters['visiting_recurring_contacts_multiplier'][0]))
           logging.debug(
               'Recurring contacts has been set to {}'.format(
                   environment.parameters['visiting_recurring_contacts_multiplier'][0]))

       if type(environment.parameters['visiting_recurring_contacts_multiplier']) == list:
           if len(environment.parameters['visiting_recurring_contacts_multiplier']) < environment.parameters['time']:
               environment.parameters['visiting_recurring_contacts_multiplier'] += [
                   environment.parameters['visiting_recurring_contacts_multiplier'][-1] for x in range(
                       len(environment.parameters['visiting_recurring_contacts_multiplier']), environment.parameters['time'])]
               logging.debug('visiting_recurring_contacts_multiplier has been lengthened by {}'.format(
                   environment.parameters['time'] - len(environment.parameters['visiting_recurring_contacts_multiplier'])))

       if kwargs.get('likelihood_awareness'):
           environment.parameters['likelihood_awareness'] = kwargs.get('likelihood_awareness')
           click.echo('Likelihood awareness has been set to {}'.format(environment.parameters['likelihood_awareness']))
           logging.debug(
               'Likelihood awareness has been set to {}'.format(environment.parameters['likelihood_awareness']))

       if kwargs.get('gathering_max_contacts'):
           environment.parameters['gathering_max_contacts'] = kwargs.get('gathering_max_contacts')
           click.echo('Max contacts has been set to {}'.format(environment.parameters['gathering_max_contacts']))
           logging.debug(
               'Max contacts has been set to {}'.format(environment.parameters['gathering_max_contacts']))

       if kwargs.get('initial_infections'):
           environment.parameters['total_initial_infections'] = round(int(kwargs.get('initial_infections')))
           click.echo('Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))
           logging.debug(
               'Initial infections have been set to {}'.format(environment.parameters['total_initial_infections']))

       # check if the stringency index has changed in the parameter file
       sringency_index_updated = False
       if environment.stringency_index != environment.parameters['stringency_index']:
           # initialise stochastic process in case stringency index has changed
           click.echo('change in stringency index detected and updated for all agents')
           environment.stringency_index = environment.parameters['stringency_index']
           if len(environment.parameters['stringency_index']) < environment.parameters['time']:
               environment.stringency_index += [environment.parameters['stringency_index'][-1] for x in range(len(
                   environment.parameters['stringency_index']), environment.parameters['time'])]

           lower, upper = -(environment.parameters['stringency_index'][0] / 100), \
                         (1 - (environment.parameters['stringency_index'][0] / 100))
           mu, sigma = 0.0, environment.parameters['private_shock_stdev']
           shocks = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                       size=len(environment.agents))
           sringency_index_updated = True

       # transform input data to general district data for simulations
       district_data = generate_district_data(environment.parameters['number_of_agents'], path=input_folder_path)

       # set scenario specific parameters
       scenario = kwargs.get('scenario', 'no-intervention')  # if no input was provided use no-intervention
       click.echo('scenario is {}'.format(scenario))
       if scenario == 'no-intervention':
           environment.parameters['likelihood_awareness'] = 0.0
           environment.parameters['visiting_recurring_contacts_multiplier'] = [
               1.0 for x in environment.parameters['visiting_recurring_contacts_multiplier']]
           environment.parameters['gathering_max_contacts'] = float('inf')
           environment.parameters['physical_distancing_multiplier'] = 1.0
           environment.parameters['informality_dummy'] = 0.0
       elif scenario == 'lockdown':
           environment.parameters['informality_dummy'] = 0.0
       elif scenario == 'ineffective-lockdown':
           environment.parameters['informality_dummy'] = 1.0

       # log parameters used after scenario called
       for param in environment.parameters:
           logging.debug('Parameter {} has the value {}'.format(param, environment.parameters[param]))

       # update agent informality based on scenario
       for i, agent in enumerate(environment.agents):
           agent.informality = what_informality(agent.district, district_data
                                               ) * environment.parameters["informality_dummy"]
           # optionally also update agent initial compliance if stringency was changed.
           if sringency_index_updated:
               agent.compliance = environment.parameters['stringency_index'][0] / 100 + shocks[i]
               agent.previous_compliance = agent.compliance

       initial_infections = pd.read_csv(os.path.join(input_folder_path, 'f_initial_cases.csv'), index_col=0)
       environment.parameters["data_output"] = kwargs.get('data_output_mode',
                                                        'csv-light')  # default output mode is csv_light

       # Simulate the model
       environment = runner(environment=environment, initial_infections=initial_infections, seed=int(seed),
                           data_folder=output_folder_path,
                           data_output=environment.parameters["data_output"])

       return environment
