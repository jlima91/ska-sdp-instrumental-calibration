# Developer Guide

## Cloning the repository

Clone the submodules with:

```bash
git clone --recurse-submodules git@gitlab.com:ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git
```

Update submodules after every pull with:

```bash
git submodule update --init
```

## Setting up virtual environment using poetry

Use Poetry to manage the Python virtual environment for development.
If Poetry is not installed, run the following command to [install a specific version of Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer):

```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.3.3 python3.11 -
```

This installs Poetry to `$HOME/.local/bin`. To make `poetry` available, add this path to the `PATH` environment variable.

```bash
export PATH=$PATH:$HOME/.local/bin
```

To permanently add this path to `PATH`, add it to your shell's login file, for example `~/.bashrc` for bash:

```bash
echo "export PATH=\$PATH:\$HOME/.local/bin" >> $HOME/.bashrc
```

Once Poetry is in `PATH`, run the following commands from the repository root:

```bash
poetry env use python3.11

poetry install
```

This will create a new virtual environment (either in current working directory or in poetry's user specific path) and install all the dependencies required for development.
For detailed package requirements, see `pyproject.toml` file.

To activate the environment, please run:

```bash
poetry shell
```

You may also want to install the shell plugin; see the [installation instructions](https://github.com/python-poetry/poetry-plugin-shell?tab=readme-ov-file#installation).

## Git hooks

To enable `git-hooks` for the current repository, run

```bash
make dev-git-hooks
```

The `pre-commit` hook is defined for the main branch and located in the `.githooks` folder. It runs the following:

1. If there is a change in `pyproject.toml`, prompt you to confirm whether the local environment has been updated.
2. Run `pylint`, which is set to fail on warnings.
3. Run `pytest`, with coverage not to fall below 80%.
4. Build documentation

A `prepare-commit-msg` hook runs after the `pre-commit` hook and formats the commit message according to the agreed convention.

A `pre-push` hook checks the GitLab CI pipeline status and warns you if the status is not "success".

> **Note:** Due to the interactive nature of these git hooks, run all git commands from a terminal. GUI-based git clients may not work correctly.

## GPG signing the commits

First, set the git username and email for your local repository.

```bash
git config user.name "username"
git config user.email "email address"
```

> The git `user.email` must match your gitlab account's email address.

Now, enable signing for commits by setting the `commit.gpgsign` config variable to `true`

```bash
git config commit.gpgsign true
```

### Signing with SSH key

To use an SSH key to sign commits, set `gpg.format` to `ssh` and `user.signingkey` to the path of your SSH public key.

```bash
PUB_KEY="path to ssh public key" # set appropriate path

git config gpg.format ssh
git config user.signingkey $PUB_KEY

# Optionally, add your ssh key added into the "allowedSignersFile"
# globally in your home/.config, so that git can trust your ssh key
mkdir -p ~/.config/git
EMAIL=$(git config --get user.email)
echo "$EMAIL $(cat $PUB_KEY)" >> ~/.config/git/allowed-signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed-signers
```

### Signing with GPG key

To use GPG keys to sign commits:

```bash
git config gpg.format openpgp
git config user.signingkey "GPG KEY" #set GPG key value
```

## Useful commands for developers

This repo contains [SKA ci-cd makefile](https://gitlab.com/ska-telescope/sdi/ska-cicd-makefile) repository as a submodule, which provides us with some standard commands out of the box.

It is **recommended** to use these instead of their simpler counterparts. For example, use `make python-test` instead of `pytest`.

Run `make help` to get a list of all supported commands. Some of the most commonly used commands are listed below:

``` bash
# Formatting the code
make python-format

# Linting checks
make python-lint

# Running tests
make python-test

# Generating source files for docs
make -C docs/ create-doc

# Building html documentation
make docs-build html

# Building oci images
make oci-build-all
```

## Making a release

### Pre-requisites

#### Update dependencies

We should ensure that the pipeline can always run with the latest version of its dependencies, unless explicitly limited due to any constraint. For this:

1. Run the `poetry update` command to update the dependencies to their latest versions.
   1. To minimize effort, you may try to only update the `main` group dependencies by explicitly passing dependency names to the above command, e.g. `poetry update dask xarray ...`
2. With these updated dependencies, ensure that the pipeline runs end-to-end, with all stages and export tasks enabled.
3. If a version upgrade causes any failure, add necessary upper limits in `pyproject.toml` and also add a backlog item to support the latest version.

#### Update make submodule

1. Run `make make` to update the `.make` submodule to the latest master commit.
2. Ensure that the `.githooks/pre-commit` hook runs successfully.

#### Update benchmarking scripts

The SKA benchmarking machinery will use the scripts and config required for benchmarking from the released version of the pipeline.
We should ensure that these are aligned with the latest changes/improvements in the pipeline. Therefore:

1. Make necessary changes to the scripts and config in `scripts/benchmark`.
2. Ensure that the scripts can run successfully with the latest version of INST.

### Release guidelines

We follow the guidelines for a release given [on this page](https://developer.skatelescope.org/en/latest/tutorial/release-management/automate-release-process.html#how-to-make-a-release).

The following steps are simplified and specific to this repository:

1. Make sure that all the changes are committed, and the local git working area is clean. Since we follow [trunk based development](https://developer.skao.int/en/latest/explanation/branching-policy.html#trunk-based-development), the active branch should be the `main` branch.

1. Check the current version using `make show-version`.

1. To bump up the release, run either one of the following commands

    ```bash
    make bump-patch-release
    make bump-minor-release
    make bump-major-release
    ```

    These commands should automatically update the version information in the following files:

    - .release
    - pyproject.toml
    - docs/src/conf.py ('release' and 'version' variable)

    If it doesn't happen automatically, please make manual changes.

    The `make show-version` command should now show the next version. Use this version for all the later changes.

1. In addition to the above files, update the `__version__` variable in the following modules:

    - `src/ska_sdp_instrumental_calibration/__init__.py`

2. Add a new `H2` heading in [CHANGELOG](CHANGELOG), and add release notes under that heading. The heading should be the new version number.

3. Using the new version, create a new issue on the [Release Management](https://jira.skatelescope.org/projects/REL/summary) Jira Project with a summary of your release, and set the issue status to "IN PROGRESS".

4. Stage all the changes, create a new commit with:

    1. The JIRA Ticket ID = ID of the release issue created in previous step.

    2. The commit message title = "Bump version to V.V.V"

5. Push the changes to the `main` branch. **Make sure that the pipeline is green.**

6. Create a git-tag for the new version using `make create-git-tag` command.

7. Run `make push-git-tag` to push the tag to the main branch.

### Post-release

The SKA software is deployed using [spack](https://spack.io/). We maintain spack packages for all of our repositories in the [ska-sdp-spack](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack) repository. After each release, ensure these packages are updated so that the latest version of the pipeline can be deployed to production. General instructions:

1. Add the new version in the [corresponding package](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/tree/main/packages/py-ska-sdp-instrumental-calibration/package.py?ref_type=heads).
2. Update the spack environment (whichever is latest) with the new version, and update the lock files.
3. Raise an MR with the above changes.

Refer to the [ska-sdp-spack documentation](https://developer.skao.int/projects/ska-sdp-spack/en/latest/PACKAGING.html#) for up-to-date instructions on updating the package.
