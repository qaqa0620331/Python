# Python standard library

## Argparse

>* https://docs.python.org/3.6/howto/argparse.html
>* https://docs.python.org/2.7/howto/argparse.html

範例:
```
```

#### argparse.ArgumentParser類別
```
class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], 
formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, 
conflict_handler='error', add_help=True, allow_abbrev=True)
```
>* Create a new ArgumentParser object. 
>* All parameters should be passed as keyword arguments. 

參數說明
>* prog - The name of the program (default: sys.argv[0])
>* usage - The string describing the program usage (default: generated from arguments added to parser)
>* description - Text to display before the argument help (default: none)
>* epilog - Text to display after the argument help (default: none)
>* parents - A list of ArgumentParser objects whose arguments should also be included
>* formatter_class - A class for customizing the help output
>* prefix_chars - The set of characters that prefix optional arguments (default: ‘-‘)
>* fromfile_prefix_chars - The set of characters that prefix files from which additional arguments should be read (default: None)
>* argument_default - The global default value for arguments (default: None)
>* conflict_handler - The strategy for resolving conflicting optionals (usually unnecessary)
>* add_help - Add a -h/--help option to the parser (default: True)
>* allow_abbrev - Allows long options to be abbreviated if the abbreviation is unambiguous. (default: True)
