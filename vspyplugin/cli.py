from __future__ import annotations

import logging
import os
import sys
from argparse import ArgumentParser
from subprocess import check_call, run

os.environ['vspyplugin_is_cli'] = 'True'


def main() -> None:
    from vspyplugin import PyBackend

    logging.basicConfig(format='{asctime}: {levelname}: {message}', style='{')
    logging.Formatter.default_msec_format = '%s.%03d'

    if sys.stdout.isatty():
        for level, string in [
            (logging.DEBUG, "\033[0;32m%s\033[0m"), (logging.INFO, "\033[1;33m%s\033[0m"),
            (logging.WARNING, "\033[1;35m%s\033[1;0m"), (logging.ERROR, "\033[1;41m%s\033[1;0m")
        ]:
            logging.addLevelName(level, string % logging.getLevelName(level))

    backend_names = [backend.name.lower() for backend in PyBackend if backend >= 0]

    parser = ArgumentParser(prog='vspyplugin')
    parser.add_argument(
        'actions', choices=['install'], type=str, default='install',
        help='Actions to perform. install: Install missing dependencies for specified backend'
    )
    parser.add_argument(
        'backend', choices=backend_names,
        nargs='?', type=str, help='What backend the dependencies should be checked for'
    )
    parser.add_argument(
        '--cuda', type=str, help='Manually specify the cuda version if it can\'t be automatically get.'
    )

    args = parser.parse_args()

    if not args.backend:
        logging.error('You must specify a backend!')
        parser.print_help()
        sys.exit(1)

    backend = PyBackend(backend_names.index(args.backend.lower()))

    if 'install' in args.actions:
        def _get_call(url: str) -> int:
            try:
                return check_call([sys.executable, '-m', 'pip', 'install', url, '-U', '--no-cache-dir'])
            except Exception:
                return 1

        for module, version in backend.dependencies.items():
            if module == 'cupy':
                if not args.cuda:
                    nvcc = run(['nvcc', '--version'], capture_output=True)

                    if nvcc.returncode != 0:
                        smi = run(['nvidia-smi', '-q'], capture_output=True)

                        if smi.returncode != 0:
                            color, message = 31, (
                                f'There was an error retrieving cuda version for {backend.name}!'
                                '\nPlease specify it with "--cuda xx.y"!'
                            )
                        else:
                            cuda_version = smi.stdout.splitlines()[5].decode().split(':')[-1].strip()
                    else:
                        cuda_version = nvcc.stdout.splitlines()[3].decode().split(
                            ',')[-2].replace('release', '').strip()
                else:
                    cuda_version = str(args.cuda)

                fver = float(cuda_version)

                if fver >= 11.2:
                    module = 'cupy-cuda11x'
                elif fver >= 10.2:
                    module = f"cupy-cuda{cuda_version.replace('.', '')}"
                else:
                    color, message = 31, f'There cuda version for {backend.name} is unsupported! ({fver})'
                    break

            mod_install = f'{module}>={version}'
            if _get_call(mod_install):
                color, message = 31, f'There was an error updating {backend.name} dependencies! ({mod_install})'
                break
        else:
            color, message = 32, f'Successfully updated {backend.name} dependencies! ({len(backend.dependencies)})'

        if sys.stdout and sys.stdout.isatty():
            message = f'\033[0;{color};1m{message}\033[0m'

        print(f'\n\t{message}\n')


if __name__ == '__main__':
    main()
