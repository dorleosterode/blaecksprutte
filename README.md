# blaecksprutte

blaecksprutte is a tagging script, that works on notmuch
databases. you have to train blaecksprutte on your notmuch database,
with already tagged emails. from this corpus blaecksprutte learns for
which emails you use which tags and tags your mails for you. for now,
blaecksprutte can only operate on the default notmuch database on your
system. configuration is not possible.

### dependencies

blaecksprutte uses scikit-learn and the notmuch python api. make sure,
you have installed both of them.

### training

to train your personal email tagger use:

```bash
python blaecksprutte.py train
```

the trained model is stored in 'tagger.pkl'.

### tagging your emails

blaecksprutte searches for all emails containing the 'new' tag and
calculates probable tags for these emails. the new tag is removed. for
tagging your emails use:

```bash
python blaecksprutte.py tag
```

you should be able to add to your notmuch command line as in for
example:

```bash
watch -n 180 "mbsync -a && notmuch new && python blaecksprutte.py tag"
```

be sure to execute this command from within the blaecksprutte
directory. else the trained model can't be found.

### contributing and improving

please contact me or write me issuse and send patches. input of any
kind is welcome :)