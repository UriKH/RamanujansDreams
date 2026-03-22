### How can I add a new Search Module?

**What do I need to do?**
1. Create a new class which inherits from `SearchMethod` similar to the one in `examples/search_method.py`.
2. Create a new class which inherits from `SearcherModScheme` similar to the one in `examples/search_module.py` .

**How?**
- In the search **method** implement the internal logic and how the search algorithm works.
- In the search **module** implement the external interface - how to use your algorithm in the context of the system? (e.g., Where and how to export data?)

**Note:** In case your implementation is not intended to be a local one and you want to contribute to the system, please put `MySearchMethod` and `MySearchMod` in different files and place `my_search_method.py` in `dreamer/search/methods/<your method's name>` and `my_search_mod.py` in `dreamer/search_stage/searchers/<your searcher's name>`.

### How can I add a new Analysis Module?

**What do I need to do?**
1. Create a new class which inherits from `AnalyzerScheme` similar to the one in `examples/analyzer.py`.
2. Create a new class which inherits from `AnalyzerModScheme` similar to the one in `examples/analysis_module.py`.

**How?**
- In the analyzer implement the internal logic and how the analysis algorithm works. 
- In the analysis module implement the external interface - how to use the analyzer? How to prioritize searchables? etc.

**Note:** In case your implementation is not intended to be a local one and you want to contribute to the system, please put `MyAnalyzer` and `MyAnalysisMod` in different files and place `my_analyzer.py` and `my_analysis_mod.py` in `dreamer/analysis/analyzers/<your analyzer's name>` (with your extra files if any).


## Developer Guide (WIP)
Globally the system looks as follows:
```
RT-CMF-sys
│   access.py
│
├─── system
├─── configs
├─── db_stage
├─── analysis_stage
├─── search_stage
└─── utils
```

### Database
| Stage                 | Path            | Description     | Example |
| :-------------:       | :-------------: | :-------------: | :-------: |
| Main folder           | `db_stage`      |  Contains a file with schemes of DB **module** and general DB + special errors file| |
| Modules               | `db_stage/DBs`  |  Each module directory contains an implemenation of the **module** in a `X_mod.py`, `config.py` and opionally an implementation of the service. | `db_v1` folder in which there are `db_mod.py`   (implementing the module DBMod class), `db.py` (implementing the DB)|
| Supported functions   |`db_stage/funcs` | All functions are used by the Formatter defined in `formatter.py`. Each inspiration function will be implemented in a file in the same folder. |            |
```
db_stage
│   db_scheme.py
│   errors.py
│
├─── DBs
│    └─── db_v1
│           config.py
│           db.py
│           db_mod.py
│
└─── funcs
        config.py
        formatter.py
        pFq_fmt.py
```

### Analysis
| Stage                 | Path            | Description     |
| :-------------:       | :-------------: | :-------------: |
| Main folder           | `analysis_stage`      | Contains a file with schemes of Analyzer **module** and general Analyzer + special errors file| 
| Modules               | `analysis_stage/analyzers`  |  See same row in _**Database**_ |
| Searchable spaces (e.g. shard)   |`analysis_stage/subspaces` | Each subspace has its own directory in which contains an implementaion of the subspace according to `searchable.py` + optional `config.py` | 
```
analysis_stage  
│    analysis_scheme.py  
│    errors.py  
│      
├─── analyzers  
│    └─── analyzer_v1  
│             analyzer.py  
│             analyzer_mod.py  
│             config.py  
│  
└─── subspaces  
     │   searchable.py
     │  
     └─── shard  
              shard.py  
              shard_extraction.py
 ```

### Searching
| Stage                 | Path            | Description     |
| :-------------:       | :-------------: | :-------------: |
| Main folder           | `search_stage`      | Contains a file with schemes of Searcher **module** and general Searcher | 
| Modules               | `search_stage/searchers`  |  See same row in _**Database**_ |
| Search methods    |`search_stage/methods` | Each method has its own directory in which contins an implementaion of the method according to `searcher_scheme.py` + optional `config.py` | 

```
search_stage
│   data_manager.py
│   searcher_scheme.py
│   errors.py
│
├─── methods
│    └─── serial
│             serial_searcher.py
│       
└─── searchers
     └─── searcher_v1
              config.py
              searcher_mod.py
```

### System and utils
- `system` contains system implementation, module implementation and system unique errors.
- `utils` contains utilities like logger and usfull type annotations for most of the code. `geometry` contains more specific utilities regarding geometrical calculations. 
```
system
    errors.py
    system.py
    module.py
utils
│   logger.py
│   types.py
│   
└─── geometry
        plane.py
        point_generator.py
        position.py
```
