# NLP_project_music_ai
NLP_2022_fall project 

Junghyun Na, Wangjiayue

# Motivation
Currently, MIDI files of each piece of performance is used as the input to the MusicBert. Based on the input MIDI files, MusicBert is fine-tuned to regress the annotation label of each piece, or to classify the player ID. We thought that classification of the player ID can be improved by also considering the annotation label during the classification. 

# Baseline performance 

We checked the baseline performance using the ```base``` checkpoint, and modified the fine-tuning script to fit our classification task

```
bash scripts/classification/train_xai_base_small.sh
```

A small change is apllied to the evaluation script to fit our task

```
python ./eval_xai_py
```

# Auxiliary classifier
We devised an auxiliary classifer to help MusicBert. It takes the annotation of each piece of performance as the input, and classifies the player ID of the performer. Same ```midi_label_map_apex_reg_cls.json``` file created from ```map_midi_to_label.py``` is used to extract the annotation data for each performance as input data, and the performer ID as the label. Various supervised classification methods (SVM, RandomForest, ...) were used. Codes for the data preprocessing and the auxiliary classifier is in ```./Auxiliary_classifier/Auxiliary_classifier.ipynb```

# Changing the loss function
We changed the loss function of the MusicBert to see if the change leads to better output, and if it fits better with the auxiliary classifiers. The changes were made in the ```./musicbert/__init__.py```. ```class LabelCrossEntropy, class LabelCrossEntropy_LSR, class OnlineLabelSmoothing, class Bootstrapping``` was implemented in the ```__init__.py``` file and the loss function in the ```class MusicBERTM2PCriterionForXAI``` was changed correspondingly.

```
...
targets = model.get_targets(sample, [logits])
#sample_size = targets.numel()
sample_size = logits.size()[0]

targets = targets[:,-1]
loss_fct = nn.CrossEntropyLoss(reduction='sum')        ----------> Point of change
loss = loss_fct(logits, targets.long())

logging_output = {
    "loss": loss.data,
    "ntokens": sample["ntokens"],
    "nsentences": sample_size,
    "sample_size": sample_size,
}
...
 ```
The checkpoint of MusicBert for each of the loss function modifications is in this [link](https://drive.google.com/drive/folders/1vED6xkJ5lOxrgP7PY8A_8B9R14Sy92ni)
