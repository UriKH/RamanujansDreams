# Ramanujan's Dreams

## Installation
For installation as a package run: `pip install git+https://github.com/UriKH/RamanujanDreams.git`  

## Usage
Interaction with the system is via the System class (`from dreamer import System`) and using the config files.
Common usage example with detailed instructions in [colab](https://colab.research.google.com/drive/1t6qo0LBBHTHTQyojXH566cNJRBhziN_3?usp=sharing).

**Note:** 
- When loading inspiration functions, you can use formerly computed CMFs using pickle files (might be unstable), maunally list the inspiration functions or using a DB (instructions below).
- Changing configurations could be done in two ways:
  1. Using `config.configure(<config_section> = {<configuration-name> : <new value>})` - that way new configurations could be added to newly developed modules.
  2. Using each section's private configuration e.g. `db_config.USAGE = DBUsage.RETRIEVE_DATA`.

### Loading using a DB
1. You can add to the DB manually (i.e. by using its interface) or by loading via a json file
2. To create a loadable json file run the following (with your inspiration functions listed):
    ```
    dreamer.loading.DBModScheme.export_future_append_to_json(
        [ <your inspiration functions> ],
        path='my_append_instruction'
    )
    ```
3. On system creation, insert the inspiration functions sources as `if_srcs=[BasicDBMod(json_path='my_append_instruction.json')]`  
   When reading this file, the system will execute the `append` command and will try to add the inspiration function ${}_2F_1(0.5)$ to set of inpiration funcitons for $\pi$ with the shift in start point as $x=0,~y=0,~z=\text{sp.Rational(1,2)}$.

(View full execution flow in the Colab link)

**(!) Please note that this package is a work in progress. Please inform / open an issue for any bug or error you encounter :)**

## Structure and Notes
### Structure:
The system is composed of 3 stages:
1. Database - storing and retrieving mapping from a constant to the inspiration functions.
2. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
3. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.

**Note:** each module could be executed independently of the others. In its current version, the system only wraps the modules together and connect them. 

### Configuration
Configuration management is done using distinct configuration files for each stage (or section - like `system`):
- `loading` - `db_config` (as only databases need configurations here)
- `analysis` - `analysis_config`
- `search` - `search_config`
- `system` - `sys_config`


Each `<X>_config` contains the configurations for this section. You can access those directly in order to view the current values.  
In order to change them you can use: `<X>_config.<property> = <new-value>`  
Or, by using the global configuration manager: `config.configure(<X> = {<property> : <new-value> })`  
The latter allows the **addition of new configurations**.
