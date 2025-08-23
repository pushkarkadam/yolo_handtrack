===================
About Documentation
===================

Sphinx documentation is used for generating the docs.

To build the sphinx document, execute the following:

.. code-block:: bash

    sphinx-build -b html docs/source/ docs/build/html

If some of the pages were added later and do not appear in the navigation bar, then using the terminal navigate to ``docs`` folder where ``make.bat`` and ``Makefile`` exists.

Run the following command:

.. code-block:: bash

    make clean

Then again go back to the root of the repository and run the sphinx document build comannd from the previous line.