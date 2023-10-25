#!/usr/bin/env bash

# INSTRUCTIONS:
#   This script will generate a test answer key in the directory you specify in the command line argument.
#   The answer key will be zipped and named based on the branch and the version of the library.
#   This should only be necessary if you are hosting the answers for github actions use.



# -- COLORS -- #
RED="\x1b[31m"
GREEN="\x1b[32m"
RST="\x1b[0m"
BLDMGT="\x1b[35;1m"
BLDRD="\x1b[31;1m"
CYANBLD="\x1b[36;1m"
CGP_TXT="[$CYANBLD Cluster Generator Dev $RST]:"
DONE="[$GREEN DONE $RST]"

# -- SETTING UP THE ENVIRONMENT -- #
# !! If you are not a UofU affiliated user hosting on CHPC, this may need to be altered.

USE_CHPC=0 # Indicate that you are using a CHPC custom python env.
REPOSITORY_URL="https://www.github.com/eliza-diggins/cluster_generator"
REPO_NAME="cluster_generator"
OUTPUT_LOCATION="$(pwd)"
TEST_BRANCH="pull-request-2"
TEST_VERSION="latest"


# Reading the user input
while getopts 'l:b:v:' OPTION; do
  case "$OPTION" in
  l)
    OUTPUT_LOCATION=$OPTARG

    if [ ! -d "$OUTPUT_LOCATION" ]; then
      printf "%b\n" "$CGP_TXT $BLDRD Invalid Location: $OUTPUT_LOCATION"
      exit 1
    fi
    ;;
  b)
    TEST_BRANCH="$OPTARG"
    ;;
  v)
    printf "%b\n" "$CGP_TXT $RED TEST_BRANCH option is not yet implemented. This will be ignored."
    ;;
  ?)
    echo "script usage: $(basename \$0) [-l path] [-b branch-name] [-v version]" >&2
    exit 1
    ;;
  esac
done

# Loading the modules for the custom python installation.
printf "%b\n" "$CGP_TXT Constructing the installation and test generation environment."

if [ $USE_CHPC == 1 ]; then
  module use $HOME/MyModules
  ml miniconda3/latest
fi

cd $OUTPUT_LOCATION

printf "%b" "$CGP_TXT Cloning repository into $OUTPUT_LOCATION..."

git clone $REPOSITORY_URL -b $TEST_BRANCH >/dev/null 2>/dev/null

if [ ! $? == 0  ]; then
  if [ ! -d "$REPO_NAME" ]; then
    printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to clone repository $REPOSITORY_URL on branch $TEST_BRANCH."
    exit 1
  else
    printf "\n%b\n" "$CGP_TXT $BLDMGT Warning$RST: There appears to be a repository already in existence. Continuing cautiously."
  fi
fi

printf "%b\n" "$DONE"

#-------------------------------------------------------------------------------------------------------#
# Installing the software
cd "$(pwd)/$REPO_NAME" || exit 1

printf "%b\n" "$CGP_TXT Installing $REPO_NAME in $(pwd)..."

pip install -e .

if [ ! $? == 0 ]; then
  printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to install $REPO_NAME."
  exit 1
fi

printf "%b\n" "$CGP_TXT Installing $REPO_NAME in $(pwd)... $DONE"

#------------------------------------------------------------------------------------------------------#
# Generating the tests
cd "$OUTPUT_LOCATION" || exit 1
printf "%b\n" "$CGP_TXT Creating directory $REPO_NAME-$TEST_BRANCH-answers to store answers."

mkdir "$REPO_NAME-$TEST_BRANCH-answers" >/dev/null 2>/dev/null

if [ ! $? == 0 ]; then
    printf "%b\n" "$CGP_TXT $BLDMGT Warning$RST: Testing directory appears to already exist. Trying to delete it and rebuild."

    rm -r "$REPO_NAME-$TEST_BRANCH-answers" >/dev/null 2>/dev/null

    if [ ! $? == 0 ]; then
        printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to delete $REPO_NAME-$TEST_BRANCH-answers."
        exit 1
    else
        mkdir "$REPO_NAME-$TEST_BRANCH-answers" >/dev/null 2>/dev/null
    fi

    if [ ! $? == 0 ]; then
        printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to delete $REPO_NAME-$TEST_BRANCH-answers."
        exit 1
    else
      printf "%b\n" "$CGP_TXT $BLDMGT Warning$RST: Succeeded in delete and rebuild of $REPO_NAME-$TEST_BRANCH-answers"
      fi
fi

printf "%b\n" "$CGP_TXT Running pytest on $REPO_NAME (BRANCH = $TEST_BRANCH)"

cd "$OUTPUT_LOCATION/$REPO_NAME" || exit 1

pytest "$REPO_NAME" --answer_dir="$OUTPUT_LOCATION/$REPO_NAME-$TEST_BRANCH-answers" --answer_store
