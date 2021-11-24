# Git and GitHub Cheatsheet

A short document detailing the basics of git and github.

## Configurating 

### Setting up details:

Configure your git username and email using the following commands.
```
$ git config --global user.name "[name]"
```
```
$ git config --global user.email "[email]"
```


### Creating Repos:

Repos are where all of your code and resources are stored.

Create a github repo in the current directory:
```
$ gh repo create [name]
```

Track a new file in the repo:
```
$ git add [file]
```

Submits changed files as being ready to be uploaded. 
```
$ git commit -m "[commentAboutChange]"
```

Upload changes that have been committed to your branch. Origin is the shorthand of the repository URL. 
```
$ git push -u origin [branch]
```

Downloads files from the repo using pull for entire repos, or fetch and switch for branches
```
$ git pull 
```
```
$ git fetch origin [remoteBranch]:[newLocalBranch]
$ git switch [newLocalBranch]
```


### Forking:

Create a fork of a repo. A fork is a copy of a repo, commonly used to propose changes to existing projects without changing it directly. This is not a branch. Branches are used to work on singular features, and can be used only by contributors.
```
$ gh repo fork JakubCzarlinski/HelloWorld 
```

Let's say the work you have done is great and you want to change the original project. A pull request is used for this. This will have to be accepted by a contributor of the repo.



### Branching:

Mangage branches of a repo using the following commands.

To create a new branch:
```
$ git branch [name]
```

To identify what branch you are checked out to:
```
$ git status
```

To switch branch:
```
$ git switch -c [new-branch] 
```
```
$ git switch [existing-branch]
```
```
$ git checkout [existing-branch]
```


### Logs:

Useful to see what changes have been made to branches and files.

To check the log of the current branch:
```
$ git log
```

To check the log of a single file:
```
$ git log --follow [file-name]
```

