### Add a search module

**What do I need to do?**
1. Create the new search method templated using as a SearchMethod
2. Create a search module which is templated using a SearcherModScheme

**How?**
- In the search method implement the internal logic and how the search works.
- In the search module implement the external interface - how to use the search method? Where and how to export data? etc.

**NOTE:** If implementation is not local and you want to contribute please put the `MySearchMethod` and `MySearchMod` in different files and place `my_search_method.py` in `dreamer/search_stage/methods/<your method's name>` (with your extra files if any) and `my_search_mod.py` in `dreamer/search_stage/searchers/<your searcher's name>` (with your extra files if any)

**Code template:** See `examples/search.py`

### Add an analysis module
WIP
