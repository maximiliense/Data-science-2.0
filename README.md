# Data science 2.0

## Best practices
Before starting your work, please create a 
[branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging).

### Personal experiments

The branch you need to create for your experiments:
```bash
git checkout -b your_branch_name
git push -u origin your_branch_name
```
Then when all updates are done, commit and pushed.
```bash
git checkout master
git merge your_branch_name
git branch -D your_branch_name
git push
git push origin --delete your_branch_name
```

Notice that you do not need to delete your branch each time, you can merge it with the master branch:
```bash
git merge origin/master
git push -u origin your_branch_name
```
or rebase it to the master level:
```bash
git fetch
git rebase origin/master
```

### Engine (and deep updates in Datascience)

The branch to create if you need to update the engine (and files related to this update).
```bash
git checkout -b engine
git push -u origin engine
```
Then when all updates are done, commit and pushed.
```bash
git checkout master
git merge engine
git branch -D engine
git push
git push origin --delete engine
```

# How to 
Please check the [wiki](https://github.com/maximiliense/Data-science-2.0/wiki).
