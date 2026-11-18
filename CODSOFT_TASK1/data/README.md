# Data

The raw dataset is **not committed** (≈105 MB). Place the extracted folder here:

```
data/raw/Genre Classification Dataset/
    train_data.txt          54,214 labelled movies  (ID ::: TITLE ::: GENRE ::: DESCRIPTION)
    test_data.txt           54,200 unlabelled movies (ID ::: TITLE ::: DESCRIPTION)
    test_data_solution.txt  54,200 official labels   (ID ::: TITLE ::: GENRE ::: DESCRIPTION)
    description.txt         format description and source
```

Source (as documented in `description.txt`): `ftp://ftp.fu-berlin.de/pub/misc/movies/database/`
(IMDb plain-text data; distributed on Kaggle as the "Genre Classification Dataset IMDb").

A different location can be used with `MOVIE_GENRE_DATA_DIR=/path/to/folder` or
`python train.py --data-dir /path/to/folder`.
