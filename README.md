# blaecksprutte

blaecksprutte is a tagging script, that works on notmuch
databases. you have to train blaecksprutte on your notmuch database,
with already tagged emails. from this corpus blaecksprutte learns for
which emails you use which tags and tags your mails for you. for now,
blaecksprutte can only operate on the default notmuch database on your
system. configuration is not possible.

### dependencies

blaecksprutte uses scikit-learn and the notmuch python api. make sure,
you have installed both of them. also the progressbar package of
python is used.

### training

to train your personal email tagger use:

```bash
python blaecksprutte.py train
```

the trained model is stored in your maildir in 'blaecksprutte.db'

### validating

you can check, how well the classifier can perform on your mails. when
you use:

```bash
python blaecksprutte.py validate
```

the classifier is trained on 0.6 of your mails and tested on the other
0.4. a classification report with precision, recall and f1-score for
every tag is printed to stdout.

### optimizing the hyperparameters

blaecksprutte searches 90 combinations of 3 hyperparameters for the
best fitting parameters on your dataset. this is done automatically
before your first training run. you can choose the best parameters
manually with:

```bash
python blaecksprutte.py optimize
```

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

### contributing and improving

please contact me or write me issuse and send patches. input of any
kind is welcome :)