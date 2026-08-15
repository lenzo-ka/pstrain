# Decision-tree and tied-state comparison

Decision-tree investigations use three different comparisons. They answer
different questions and must not be substituted for one another.

1. **Literal tree equality** compares rooted, ordered topology and the exact
   question expression at each internal node after normalizing whitespace.
   Node numbers, file order, likelihood gains, and occupancies are ignored.
   Equality supports a claim that these serialized decision structures match.
   Difference does not by itself show a different tied-state allocation.
2. **Partition equality** compares the equivalence classes of context-state
   rows within each base-phone/emitting-state subject. Senone IDs are ignored;
   only which contexts share a senone matters. Equality supports an allocation
   claim. Pair disagreements quantify allocation differences.
3. **Keyed-row agreement** compares the numeric senone ID on corresponding
   context-state rows. It is label-sensitive: an ID permutation can reduce the
   count while leaving every partition unchanged. It is a serialization
   diagnostic, not an allocation metric.

Run [`scripts/tree_compare.py`](../../scripts/tree_compare.py) with
`--mode literal-tree` (the default), `--mode partition`, or
`--mode keyed-row`. Keeping the modes explicit prevents a literal or
label-sensitive result from being reported as partition evidence.

The literal comparator deliberately does not compare score or occupancy
fields (including values used as pruning thresholds). It also does not decide
whether two different question expressions are semantically equivalent; it
compares their normalized expression text exactly.
