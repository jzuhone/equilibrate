#!/usr/bin/env bash
#======================================================================================================================#
# CONFIGURATION
RED="\x1b[31m"
GREEN="\x1b[32m"
RST="\x1b[0m"
BLDMGT="\x1b[35;1m"
BLDRD="\x1b[31;1m"
CYANBLD="\x1b[36;1m"
CGP_TXT="[$CYANBLD Cluster Generator Dev $RST]:"
DONE="[$GREEN DONE $RST]"
#======================================================================================================================#
# USER SETTINGS --> These should be altered specific to the developer's need
USE_CHPC=0 # If 1, the custom CHPC installation will be used.
OUTPUT_DIRECTORY="$(pwd)" # The directory into which the answers should be placed

# BUILD Settings #
REPOSITORY_URL="https://www.github.com/eliza-diggins/cluster_generator" # Used for installation
BRANCH="master" # Overridden by -b tag.
TEST_TYPE="all" # 'all' or 'core'.
REPO_NAME="cluster_generator"

#======================================================================================================================#
# Managing command line inputs and setting up.

# ***************************************** #
#  AVAILABLE COMMAND LINE ARGS
#  -b: determine the branch of the master repository to grab from.
#  -r: The repository url to use.
#  -t: The testing type to use
#  -l: The output directory.
# ****************************************** #

while getopts 'l:b:r:t:' OPTION; do
  case "$OPTION" in
  l)
    OUTPUT_DIRECTORY="$OPTARG"

    if [ ! -d "$OUTPUT_DIRECTORY" ]; then
      printf "%b\n" "$CGP_TXT $BLDRD Invalid Location: $OUTPUT_DIRECTORY"
      exit 1
    fi
    ;;
  b)
    BRANCH="$OPTARG"
    ;;
  r)
    REPOSITORY_URL="$OPTARG"
    ;;
  t)
    TEST_TYPE="$OPTARG"
    ;;
  ?)
    echo "script usage: $(basename \$0) [-l path] [-b branch-name] [-r repository URL] [-t test types]" >&2
    exit 1
    ;;
  esac
done

#======================================================================================================================#
# Preparing the environment

printf "%b\n" "$CGP_TXT Runtime environment prepared with the following runtime variables:"
printf "\t \t %-30b \t \t %-12b\n" "$BLDMGT VARIABLE$RST" "$BLDMGT VALUE$RST"
printf "\t \t %-30b \t \t %-12b\n" "$BLDMGT --------$RST" "$BLDMGT -----$RST"
printf "\t \t %-30b \t \t %-12b\n" "$CYANBLD BRANCH$RST" "$CYANBLD $BRANCH$RST"
printf "\t \t %-30b \t \t %-12b\n" "$CYANBLD REPOSITORY$RST" "$CYANBLD $REPOSITORY_URL $RST"
printf "\t \t %-30b \t \t %-12b\n" "$CYANBLD REPO-NAME$RST" "$CYANBLD $REPO_NAME $RST"
printf "\t \t %-30b \t \t %-12b\n" "$CYANBLD TEST-TYPES$RST" "$CYANBLD $TEST_TYPE $RST"
printf "\t \t %-30b \t \t %-12b\n" "$CYANBLD OUTPUT_DIRECTORY$RST" "$CYANBLD $OUTPUT_DIRECTORY $RST"
printf "%s\n" "############################################################################################################"

if [ $USE_CHPC == 1 ]; then
  module use $HOME/MyModules
  ml miniconda3/latest
fi

cd $OUTPUT_DIRECTORY

printf "%b" "$CGP_TXT Cloning repository into $OUTPUT_DIRECTORY..."

git clone $REPOSITORY_URL -b $BRANCH >/dev/null 2>/dev/null

if [ ! $? == 0  ]; then
  if [ ! -d "$REPO_NAME" ]; then
    printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to clone repository $REPOSITORY_URL on branch $BRANCH."
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
cd "$OUTPUT_DIRECTORY" || exit 1
printf "%b\n" "$CGP_TXT Creating directory $REPO_NAME-$BRANCH-answers to store answers."

mkdir "$REPO_NAME-$BRANCH-answers" >/dev/null 2>/dev/null

if [ ! $? == 0 ]; then
    printf "%b\n" "$CGP_TXT $BLDMGT Warning$RST: Testing directory appears to already exist. Trying to delete it and rebuild."

    rm -r "$REPO_NAME-$BRANCH-answers" >/dev/null 2>/dev/null

    if [ ! $? == 0 ]; then
        printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to delete $REPO_NAME-$BRANCH-answers."
        exit 1
    else
        mkdir "$REPO_NAME-$BRANCH-answers" >/dev/null 2>/dev/null
    fi

    if [ ! $? == 0 ]; then
        printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Failed to delete $REPO_NAME-$BRANCH-answers."
        exit 1
    else
      printf "%b\n" "$CGP_TXT $BLDMGT Warning$RST: Succeeded in delete and rebuild of $REPO_NAME-$BRANCH-answers"
      fi
fi

printf "%b\n" "$CGP_TXT Running pytest on $REPO_NAME (BRANCH = $BRANCH)"

cd "$OUTPUT_DIRECTORY/$REPO_NAME" || exit 1

pytest "$REPO_NAME" --answer_dir="$OUTPUT_DIRECTORY/$REPO_NAME-$BRANCH-answers" --answer_store

if [ ! $? == 0 ]; then
    printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Tests did not succeed."
    exit 1
fi

printf "%b\n" "$CGP_TXT Tests completed. All tests passed."
printf "%b" "$CGP_TXT Packaging test answers..."

cd "$OUTPUT_DIRECTORY" || exit 1
if [ -f "$REPO_NAME-$BRANCH-answers.tar.gz" ]; then
      printf "%b\n" "$CGP_TXT $BLDMGT Warning$RST: Zip archive already exists."

      rm "$REPO_NAME-$BRANCH-answers.tar.gz"
fi

tar -czvf "$REPO_NAME-$BRANCH-answers.tar.gz" "$REPO_NAME-$BRANCH-answers" >/dev/null 2>/dev/null

if [ ! $? == 0 ]; then
    printf "\n%b\n" "$CGP_TXT $BLDRD ERROR: Tar zipping failed"
    exit 1
fi

printf "%b\n" "$DONE"

exit 0
