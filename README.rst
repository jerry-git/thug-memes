Thug Memes
==========
|build|
|pypi|
|python_versions|
|codecov|

Command line Thug Meme generator written in Python.

Installation
------------

Requirements
^^^^^^^^^^^^
 - Python 3.4+

.. code:: bash

    pip3 install thug-memes

This installs the core version which supports `opencv <https://pypi.python.org/pypi/opencv-python>`__ haarcascade based detector. If you are serious with your thug memes (like you definitely should), there is also support for 
`dlib <http://dlib.net/>`__ based detector which, in general, provides better results. If you want to enjoy dlib's accuracy, please follow `dlib's own installation instructions <https://pypi.python.org/pypi/dlib>`__. Some additional dlib installation guides for macOS and Ubuntu can be found in: `[1] <https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf>`__  `[2] <https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/>`__ 

Usage
-----
.. code:: bash

    thug path/to/the/original/image 'JUST CASUALLY LIFTING' '20KGs HERE'

Will store something like this in your current directory:

|img1|

If you have installed dlib and want to use it instead of opencv detector, add option:

.. code:: bash

    --detector dlib

If you want to see the results of the detection, add *--debug* option:
 

Opencv (*--debug*):

|img2|

Dlib (*--detector dlib --debug*):

|img3|

Almost everything is configurable. You can see the used configuration with: 

.. code:: bash

    --show-config

The default configuration is in `src/thug/defaul.conf`. You can override this by defining environment variable `THUG_CONF` and pointing it to a file which contains overrides. In addition, you can override any of the configuration variables from command line with *--override* or *-o* options. For example:

.. code:: bash

    thug path/to/original/img.jpg 'WE HAVE HUGE CIGARS' 'BUT ALSO PINK TEXT' --detector dlib -o cigar_length 2 -o glasses_width 5 -o font_bgr '[180,105,255]'

|img4|

For all available options, see:

.. code:: bash

    thug --help



If you don't want to have awesome Thug elements in your meme, there is also 'a plain meme' alternative:

.. code:: bash

    meme path/to/the/original/image 'THIS IS A NORMAL MEME' 'WITHOUT THUG STUFF :(' -o font_bgr '[255,255,255]'

|img5|

Example images are from `pexels <https://www.pexels.com/photo-license/>`__.


.. |pypi| image:: https://img.shields.io/pypi/v/thug-memes.svg
   :target: https://pypi.python.org/pypi/thug-memes


.. |build| image:: https://travis-ci.org/jerry-git/thug-memes.svg?branch=master
   :target: https://travis-ci.org/jerry-git/thug-memes

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/thug-memes.svg
   :target: https://pypi.python.org/pypi/thug-memes

.. |codecov| image:: https://codecov.io/gh/jerry-git/thug-memes/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/jerry-git/thug-memes


.. |img1| image:: https://raw.githubusercontent.com/jerry-git/thug-memes/master/doc/examples/1_face_out_thug.jpg
	:height: 600pt

.. |img2| image:: https://raw.githubusercontent.com/jerry-git/thug-memes/master/doc/examples/1_face_debug_opencv.jpg
	:height: 600pt

.. |img3| image:: https://raw.githubusercontent.com/jerry-git/thug-memes/master/doc/examples/1_face_debug_dlib.jpg
	:height: 600pt

.. |img4| image:: https://raw.githubusercontent.com/jerry-git/thug-memes/master/doc/examples/3_faces_thug_custom.jpeg
	:width: 600pt

.. |img5| image:: https://raw.githubusercontent.com/jerry-git/thug-memes/master/doc/examples/normal_meme_out.jpg
	:width: 600pt