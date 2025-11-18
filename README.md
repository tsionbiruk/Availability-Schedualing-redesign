**Project Overview**

This project documents a redesign of availability and scheduling workflows for Care Diem Tours. The aim is to simulate different scheduling strategies, evaluate their performance, and generate output data to inform decision-making. The repository includes simulation code, test strategies, and output data.

**Repository Structure**
-** Scheduling/:** Contains the core simulation scripts, functions and modules for modelling availability and scheduling.

- **Strategy_test/:** Contains test cases, scenario definitions, and scripts to compare different scheduling strategies.

- **Availability-Scheduling Redesign.pdf:** The initial project documentation, including problem statement, stakeholder requirements, design options, and assumptions.

**Simulation Details**

**Scheduling logic:** The simulation models the availability of resources (tours, guides, vehicles, etc.) and the scheduling of tasks/events relative to demand and constraints.

**Strategies implemented:** Includes a baseline (current workflow) and one or more redesigned strategies (e.g., priority scheduling, rolling availability blocks, dynamic reallocation).

**Performance metrics:** Key measures such as utilisation rate, task completion rate, idle/resource wait time, and unmet demand are recorded for each strategy.

**Output data & analysis:** The scripts dump CSV/JSON of results and produce visualisations that compare strategy performance side-by-side.
