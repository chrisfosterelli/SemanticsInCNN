Semantics in CNNs
=================

This is the code for the paper "Interpreting the Information in Hidden 
Representations of Convolutional Neural Networks".

Adversarial Attacks
===================

The adversarial sections of the paper consist of three files: `adversarial.py`, 
`one_vs_two_adversarial.py`, and `adversarial_graphs.py`.

The source images are expected to be a directory containing a directory 
corresponding to the class label, with source images of that class label inside
that the attacks will be generated for. For example, here is a directory layout:

```bash
/AdversarialImages/
/AdversarialImages/accordion/
/AdversarialImages/accordion/ILSVRC2012_val_00004424.JPEG
/AdversarialImages/accordion/ILSVRC2012_val_00009350.JPEG
...
```

The map file is a pickle file containing the source classes along with the
target classes to generate the adversarial attacks for. The target classes 
should include the correlation for the target class with the source class. The
structure is as follows:

```bash
[
  [ 'accordion', ... ],
  [ 
    [ ('dog', 0.0370), ('cat', 0.0839), ... ],
    ...
  ]
]
```

The baseline concepts directory should follow the same format as the source 
images, but should contain class labels that have no overlap with either the
source or target classes for the adversarial attack.

To generate the adversarial examples, run the following script:

```bash
> python adversarial.py \
    <location of source images> \
    <map file containing target classes> \
    --classes <one or more of the source classes to select (optional)> \
    --output-images <directory to write adversarial images (optional)>
```

The generate the 1 vs. 2 results, run `one_vs_two_adversarial.py`:

```bash
> python one_vs_two_adversarial.py \
    <location of the adversarial outputs from the last step> \
    <location of baseline concepts to compare against> \
    <map file containing target classes>
```

To generate the graphics for the results in the paper, run 
`adversarial_graphs.py`:

```bash
> python adversarial_graphs.py
```
