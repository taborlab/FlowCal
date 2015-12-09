How to Contribute
=================
If you are interested in contributing to this project, either by writing code, correcting a bug, or adding a new feature, we would love your help! Below we provide some guidelines on how to contribute.

``FlowCal`` Installation for Developers
---------------------------------------
Regardless of your OS version, we recommend using ``virtualenv`` for development. A short primer on ``virtualenv`` can be found at http://docs.python-guide.org/en/latest/dev/virtualenvs/.

The recommended way to install ``FlowCal`` for development is to run ``python setup.py develop``. This will install ``FlowCal`` in a special "developer" mode. In this mode, a link pointing to the repo is made in the python installation directory, allowing you to import ``FlowCal`` from any python script, while at the same time being able to modify the repo code and immediately see the modifications.

Version Control
---------------
``FlowCal`` uses ``git`` for version control. We try to follow the `git-flow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. Please familiarize yourself with such model before contributing. A quick summary of relevant branches is given below.

* ``master`` is only used for final release versions. **Do not** directly commit to ``master``, ever.
* ``develop`` holds unreleased features, which will eventually be released into ``master``.
* *Feature branches* are branches derived from ``develop``, in which new features are committed. When the feature is completed, a merge request towards ``develop`` should be made.

Recommended Workflow
--------------------
A recommended workflow for contributing to ``FlowCal`` is as follows:

1. Report your intended change in the issue tracker on ``github``. If reporting a bug, please be as detailed as possible and try to include the necessary steps to reproduce the problem. If suggesting a feature, indicate if you're willing to write the code for it.
2. Assuming that you decided to write code, clone the repo in your computer. You can use the command ``git clone https://github.com/taborlab/FlowCal`` if you are using the command-line version of ``git``.
3. Switch to the develop branch, using ``git checkout develop``.
4. Create a new feature branch, using ``git checkout -b <feature_name>``.
5. Set up your virtual environment, if desired.
6. Install ``FlowCal`` in developer mode, using ``python setup.py develop``.
7. Write/test code, commit. Repeat until feature is fully implemented.
8. Push and submit a merge request towards ``develop``.