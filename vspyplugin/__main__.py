import os
os.environ['vspyplugin_is_cli'] = 'True'

from .cli import main

main()
