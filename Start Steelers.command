#!/bin/zsh
cd -- "${0:A:h}"
if ! command -v python3 >/dev/null 2>&1; then
    print "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/macos/"
    read "?Press Return to close."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    print "This app needs Python 3.10 or newer. Install it from https://www.python.org/downloads/macos/"
    read "?Press Return to close."
    exit 1
fi
python3 app.py
if [[ $? -ne 0 ]]; then
    read "?The app could not start. Press Return to close."
fi
