find . -type d -regextype posix-extended -iregex ".*/simulator_train_test.*" -exec rm -r {} \;
find . -type d -regextype posix-extended -iregex ".*/__pycache__" -exec rm -r {} \;

find . -type d -regextype posix-extended -iregex ".*/competition_data" -exec rm -r {} \;
find . -type f -name "*.png" -exec rm {} \;
