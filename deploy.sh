#!/bin/sh

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi

printf "\033[0;32mCommit and push static blog source...\033[0m\n"

# Add changes to git.
git add . -A
git commit -m "$msg"

# Push source and build repos.
git push origin master

# Go To Public folder
cd public

printf "\033[0;32mCommit and push public...\033[0m\n"

# Add changes to git.
git add . -A
git commit -m "$msg"

# Push source and build repos.
git push origin master
