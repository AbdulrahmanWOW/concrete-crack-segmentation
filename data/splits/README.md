# Splits

`dev.txt` and `test.txt` list the image IDs of the development and test sets
(70 / 30, seed 42). They are created once by `src.data.make_split` and committed so the
test set stays frozen. Never tune on `test.txt`.
