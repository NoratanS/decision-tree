# Decision Tree (C4.5)

Learning a decision tree from a decision table and using it to classify new cases, following ID3 / C4.5 as described in the course script (Kozak, "Systemy uczące się") and its source (Herman-Iżycka & Strzelczak, MIMUW).

## Data

**Decision Table**:
A table of cases where every column but the last is a conditional attribute and the last column is the decision attribute. (PL: tabela decyzyjna)
_Avoid_: dataset, data frame, training set (when meaning the table itself)

**Case**:
One row of a decision table: a value for each conditional attribute plus a decision. (PL: obiekt, x₁, x₂, …)
_Avoid_: record, sample, instance, row, example

**Conditional Attribute**:
A column that describes a case and may be tested in a node. (PL: atrybut warunkowy)
_Avoid_: feature, non-categorical attribute, input

**Decision Attribute**:
The single column, always last, holding the class each case belongs to. (PL: decyzja, atrybut decyzyjny)
_Avoid_: target, label, categorical attribute

**Decision Class**:
One value of the decision attribute, and the set of cases that have it. (PL: klasa decyzyjna)
_Avoid_: category, label, outcome

**Discrete Attribute**:
A conditional attribute whose values are treated as unordered names; a test on it has one branch per value.
_Avoid_: categorical, nominal

**Continuous Attribute**:
A conditional attribute whose values are ordered numbers; a test on it compares against a threshold and has two branches.
_Avoid_: numeric, real-valued

**Missing Value**:
A cell with no known value for a case.
_Avoid_: null, NaN, unknown value

## Training and test

**Training Set**:
The decision table a tree is built from.

**Test Set**:
A decision table with known decisions, used only to measure how well a tree classifies.
_Avoid_: validation set

**Experiment**:
One run of building a tree from a Training Set, optionally pruning it, and evaluating each tree on the Training Set and an optional Test Set (a separate table or a hold-out split of the Training Set).
_Avoid_: run, pipeline, session

## Split measures

**Entropy**:
The information carried by a probability distribution, I(P) = −Σ pᵢ·log₂ pᵢ. (PL: entropia)

**Info**:
Entropy of the decision classes in a set of cases, Info(T); or, for an attribute X, the weighted average of Info over the subsets X splits T into, Info(X,T). (PL: funkcja informacji)

**Gain**:
How much knowing attribute X reduces Info: Gain(X,T) = Info(T) − Info(X,T). The ID3 split measure. (PL: przyrost informacji)
_Avoid_: information gain (fine in prose, but use Gain as the name)

**Split Info**:
Entropy of the partition of T induced by attribute X itself, ignoring decision classes.

**Gain Ratio**:
Gain(X,T) / SplitInfo(X,T); corrects Gain's bias toward attributes with many values. The C4.5 split measure. (PL: zrównoważony przyrost informacji)

## Tree

**Node**:
A point in the tree that tests one conditional attribute and has one branch per outcome.
_Avoid_: split, decision node

**Leaf**:
A point in the tree that assigns a decision class.
_Avoid_: terminal node

**Threshold**:
The cut value of a test on a continuous attribute; the branches are ≤ threshold and > threshold.
_Avoid_: split point, cut point

**Case Weight**:
How much of a case is present at a point in the tree; 1 for a whole case, a fraction when a case with a missing value is spread across several branches.
_Avoid_: sample weight, fractional case

**Class Distribution**:
The weight of each decision class among the cases reaching a leaf, normalised to probabilities when classifying.
_Avoid_: class probabilities, scores

**Pruning**:
Replacing a subtree with a leaf when its estimated error on unseen cases is no lower than the leaf's. (PL: przycinanie)

**Confidence Level**:
The parameter of the pessimistic error estimate used in pruning; lower values prune harder.
_Avoid_: CF (fine as a CLI flag name only)
