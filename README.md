# Ramanujan's Dreams
Ramanujan's Dreams is a modular system for advanced search in CMFs.

## Installation
* This project is supported fully only on Mac-OS and Linux.  
If you are a Windows user, it is recommended to use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).
* Install via: `pip install git+https://github.com/UriKH/RamanujansDreams.git`

## Usage
Interaction with the system is via the System class (`from dreamer import System`) and using the config files.
Common usage example with detailed instructions in [colab](https://colab.research.google.com/drive/1t6qo0LBBHTHTQyojXH566cNJRBhziN_3?usp=sharing).  
**Note**: The Colab might be slow and unstable as it's running online. For stable run download the colab as a Jupyter notebook.

### Structure:
The system is composed of 4 stages:
1. Loading - storing and retrieving mapping from a constant to the inspiration functions.
2. Extraction - extraction of the searchables from the CMF of the inspiration functions.
3. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
4. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.

[//]: # (**Note:** each module could be executed independently of the others. In its current version, the system only wraps the modules together and connects them. )

### Configuration
Configuration management is done using distinct configuration categories which are all accessed via a global configuration manager:
```python
from dreamer import config

# Access different categories of configurations
config.extraction.<CONFIG> 
config.analysis.<CONFIG> 
config.search.<CONFIG>
config.system.<CONFIG>
config.logging.<CONFIG>
config.database.<CONFIG>

# change specific configurations
config.configure(
    <CATEGORY> = {<CONFIG>: <VALUE>, ...},
    <CATEGORY> = {<CONFIG>: <VALUE>, ...},
    ...
)

# Checkout possible configurations using the terminal 
config.<CATEGORY>.display()
```

There are a few important configurations you might want to change:
- `config.search.NUM_TRAJECTORIES_FROM_DIM` - a lambda function of the form `lambda dim: int(...)` which computes the number of trajectories to be generated from a given dimension.
- `config.analysis.NUM_TRAJECTORIES_FROM_DIM` - same configuration as above but for analysis stage.
- `config.analysis.IDENTIFY_THRESHOLD` - "what fraction of the shard was identified as containing the constant?"

[//]: # (Each `<X>_config` contains the configurations for this section. You can access those directly in order to view the current values.  )
[//]: # (In order to change them you can use: `<X>_config.<property> = <new-value>`  )
[//]: # (Or, by using the global configuration manager: `config.configure&#40;<X> = {<property> : <new-value> }&#41;`  )
[//]: # (The latter allows the **addition of new configurations**.)

### Run
A classic run would look something like this:
```python
from dreamer import System, config, log
from dreamer import analysis, search, extraction, loading

# Optional reconfigure
config.configure(...)

my_system = System(
    if_srcs=[loading.pFq(log(2), 2, 1, -1)],            # Set up the loading stage - provide inspiration functions
    extractor=extraction.extractor.ShardExtractorMod,   # Choose an extraction module
    analyzers=[analysis.AnalyzerModV1],                 # Choose an analysis module(s)
    searcher=search.SearcherModV1                       # Choose the search module
)

my_system.run(constants=[log(2)])
```

Advanced options are:
* Using a database as on of the inspiration functions source.
* Using pickled inspiration function objects from past runs as inspiration functions source.
* Using pickled past analysis results as input to the analysis stage.

### Terminal Setup

If you are a PyCharm user, the output might look a bit off due to `tqdm` default configurations.  
To make sure the output console looks right:
1. Enter: `Run > Edit Configurations > Modify Options`
2. Select: `Emulate terminal in output console`

[//]: # (#### Notes: )
[//]: # (- When loading inspiration functions, you can use formerly computed CMFs using pickle files &#40;might be unstable&#41;, maunally list the inspiration functions or using a DB &#40;instructions below&#41;.)
[//]: # (- Changing configurations could be done in two ways:)
[//]: # (  1. Using `config.configure&#40;<config_section> = {<configuration-name> : <new value>}&#41;` - that way new configurations could be added to newly developed modules.)
[//]: # (  2. Using each section's private configuration e.g. `db_config.USAGE = DBUsage.RETRIEVE_DATA`.)
[//]: # (  3. If you are a PyCharm user, your terminal might be a bit off due to `tqdm` defualt configurations.  )
[//]: # (   To make sure the terminal looks right set: `Run > Edit Configurations > Emulate terminal in output console`)
[//]: # (### Loading using a DB)
[//]: # (1. You can add to the DB manually &#40;i.e. by using its interface&#41; or by loading via a json file)
[//]: # (2. To create a loadable json file run the following &#40;with your inspiration functions listed&#41;:)
[//]: # (    ```)
[//]: # (    dreamer.loading.DBModScheme.export_future_append_to_json&#40;)
[//]: # (        [ <your inspiration functions> ],)
[//]: # (        path='my_append_instruction')
[//]: # (    &#41;)
[//]: # (    ```)
[//]: # (3. On system creation, insert the inspiration functions sources as `if_srcs=[BasicDBMod&#40;json_path='my_append_instruction.json'&#41;]`  )
[//]: # (   When reading this file, the system will execute the `append` command and will try to add the inspiration function ${}_2F_1&#40;0.5&#41;$ to set of inpiration funcitons for $\pi$ with the shift in start point as $x=0,~y=0,~z=\text{sp.Rational&#40;1,2&#41;}$.)

## License
This project is licensed under the terms of the [MIT License](LICENSE).

## Contribution
* Please open an issue for any bug or error you encounter.
* For further details see [instructions](CONTRIBUTING.md).
