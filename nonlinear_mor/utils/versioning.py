from nonlinear_mor.utils.logger import getLogger


def get_git_hash():
    """Get the current Git hash of this package.

    Returns
    -------
    Git hash, if available.
    """
    logger = getLogger('nonlinear_mor.versioning.get_git_hash')

    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        logger.info("Extracted Git hash.")
    except (ModuleNotFoundError, git.exc.InvalidGitRepositoryError):
        sha = "No Git repository found."
        logger.warning("Could not extract Git hash.")
    return sha


def get_version(package):
    """Get the current version of the provided package.

    Returns
    -------
    Package version, if available.
    """
    logger = getLogger('nonlinear_mor.versioning.get_version')

    try:
        import os
        import subprocess
        version = str(subprocess.check_output(('git', '-C', os.path.dirname(package.__file__),
                                               'log', '-1', '--format=%H %cd')))
        logger.info(f"Extracted version of package '{package.__name__}'.")
    except subprocess.CalledProcessError:
        try:
            version = package.__version__
            logger.info(f"Extracted version of package '{package.__name__}'.")
        except AttributeError:
            version = f"No version for package '{package.__name__}' found."
            logger.warning(f"Could not extract version of package '{package.__name__}'.")
    return version
