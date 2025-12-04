# Ramanujan's Dreams

## Installation
For installation as a package run: `pip install git+https://github.com/UriKH/RamanujanDreams.git`  

## Usage
Interaction with the system is via the System class (`from dreamer import System`) and using the config files.
Usage example in [colab](https://colab.research.google.com/drive/1t6qo0LBBHTHTQyojXH566cNJRBhziN_3?usp=sharing).

**Note:** 
- The names of the constants should be writen as used in `sympy` (pi - `sp.pi`, E - `sp.E`, etc.).
- If you don't want to load or execute commands using a JSON file, `json_path` could be omitted from the arguments.
- Changing configurations could be done in two ways:
  1. Using `config.configure(<config_section> = {<configuration-name> : <new value>})` - that way new configurations could be added to newly developed modules.
  2. Using each section's private configuration e.g. `db_config.USAGE = DBUsage.RETRIEVE_DATA`.

**(!) Please note that this package is a work in progress and thus not fully tested.  
Therefore, please inform / open an issue for any bug or error you encounter :)**

##### data.json
When reading this file, the system will execute the `append` command and will try to add the inspiration function ${}_2F_1(0.5)$ to set of inpiration funcitons for $\pi$ with the shift in start point as $x=0,~y=0,~z=\text{sp.Rational(1,2)}$.
```
{
    "command": "append",
    "data": [
        {
            "constant": "pi",
            "data": {
                "type": "pFq_formatter",
                "data": { "p": 2, "q": 1, "z": "1/2", "shifts": [0, 0, "1/2"] }
            }
        }
    ]
}
```

## Structure and Notes
### Structure:
The system is composed of 3 stages:
1. Database - storing and retrieving mapping from a constant to the inspiration functions.
2. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
3. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.

### Configuration
Configuration management is done using distinct configuration files for each stage (or section - like `system`):
- `database` - `db_config`
- `analysis` - `analysis_config`
- `search` - `search_config`
- `system` - `sys_config`


Each `<X>_config` contains the configurations for this section. You can access those directly in order to view the current values.  
In order to change them you can use: `<X>_config.<property> = <new-value>`  
Or, by using the global configuration manager: `config.configure(<X> = {<property> : <new-value> })`  
The latter allows the **addition of new configurations**.
