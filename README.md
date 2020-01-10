# Ride-Hailing Platform with Simulator
These codes are part of course project completed with [zhengjilai](https://github.com/zhengjilai) and [Lsc-cs](https://github.com/Lsc-cs) aiming to provide a Ride-Hailing Platform with simulator. Concretely, for orders dispatching, we constructs a bipartite graph for orders and drivers with weights of edges calculated by a value map processed from historical data using dynamic programming (inspired by [Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach](https://dl.acm.org/doi/10.1145/3219819.3219824)), and solves the matching problem by KM algorithm. Fleet management algorithm is also based on the value map.

The simulator codes are modified from [here](https://github.com/illidanlab/Simulator). We use orders data from DiDi ([Nov 2016, Xi’an City Second Ring Road Regional Trajectory Data Set](https://outreach.didichuxing.com/research/opendata/)).



## Dependencies
- Python 3

## Usage
1. Process data.  
`python data_processing/make_data.py`
2. Run simulator.  
`python run/run_fix_drivers_number_with_value_map_with_fleetmanagement.py`
