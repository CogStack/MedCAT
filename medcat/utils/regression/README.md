# Regression with MedCAT

We often end up creating new models when a new version of an ontology (e.g SNOMED-CT) comes out.
However, it is not always clear whether the new model is comparable to the old one.
To solve this, we've developed a regression suite system.

The idea is that we can define a small set of patient records with different placeholders for different findings or disorders, or anything in the ontology, really.
And we can then specify the concepts we think should fit in this patient record.

An example patient record with placeholders (the simple one from the default regression suite):
```
The patient presents with [FINDING1] and [FINDING2]. These findings are suggestive of [DISORDER].
Further diagnostic evaluation and investigations are required to confirm the diagnosis.
```
As we can see, there are three different placeholders in here: `[FINDING1]`, `[FINDING2]`, and `[DISORDER]`.
Each can be replaced with a specific name of a specific concept.
For instance, we've specified the following:
 - `[FINDING1]` -> '49727002' (cough)
 - `[FINDING2]` -> '267036007' (shortness of breath)
 - `[DISORDER]` -> '195967001' (asthma)

So with these swapped into the original patient record we get:
```
The patient presents with cough and shortness of bre. These findings are suggestive of asthma.
Further diagnostic evaluation and investigations are required to confirm the diagnosis.
```

# Using regression suite

The easiest way to use the regression suite is to use the built in endpoint:
```
python -m medcat.utils.regression.regression_checker <model pack name> [regression suite YAML]
```
While you need to specify a model pack, you do not need to specify a regression suite since the default one can be used instead.

This will first read the regression suite from the YAML, then load the model pack, and finally run the regression suite.

<details><summary>The output can look like this</summary>
Output on the 2024-06 SNOMED-CT model on the first case in the default regression suite.

```
$ python -m medcat.utils.regression.regression_checker models/Snomed2024-06-gstt-trained_ae5b08e0fb5310b2.zip
Loading RegressionChecker from yaml: configs/default_regression_tests.yml
Loading model pack from file: models/Snomed2024-06-gstt-trained_ae5b08e0fb5310b2.zip
Checking the current status
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s]
A total of 1 parts were kept track of within the group "ALL".
And a total of 756 (sub)cases were checked.
At the strictness level of Strictness.NORMAL (allowing ['FOUND_ANY_CHILD', 'BIGGER_SPAN_LEFT', 'SMALLER_SPAN', 'PARTIAL_OVERLAP', 'BIGGER_SPAN_BOTH', 'BIGGER_SPAN_RIGHT', 'FOUND_CHILD_PARTIAL', 'IDENTICAL']):
The number of total successful (sub) cases: 737 (97.49%)
The number of total failing (sub) cases   : 19 ( 2.51%)
IDENTICAL               :       730 (96.56%)
SMALLER_SPAN            :         2 ( 0.26%)
FOUND_ANY_CHILD         :         5 ( 0.66%)
FAIL                    :        19 ( 2.51%)
	Tested 'test-case-1' for a total of 756 cases:
		IDENTICAL               :       730 (96.56%)
		SMALLER_SPAN            :         2 ( 0.26%)
		FOUND_ANY_CHILD         :         5 ( 0.66%)
		FAIL                    :        19 ( 2.51%)
		Examples at Strictness.STRICTEST strictness
		With phrase: 'Description: Acute appendicitis\nCC: abdo [277 chars] d Nausea. He denied Diarrhea.\n'
			FOUND_ANY_CHILD for placeholder [FINDING1] with CUI '21522001' and name 'abdominal colic'
		With phrase: 'Description: Acute appendicitis\nCC: [FIN [273 chars] d Nausea. He denied Diarrhea.\n'
			SMALLER_SPAN for placeholder [FINDING1] with CUI '21522001' and name 'abdomen colic'
		With phrase: 'Description: Acute appendicitis\nCC: abdo [273 chars] d Nausea. He denied Diarrhea.\n'
			SMALLER_SPAN for placeholder [FINDING1] with CUI '21522001' and name 'abdomen colic'
		With phrase: 'Description: Acute appendicitis\nCC: abdo [293 chars] d Nausea. He denied Diarrhea.\n'
			FOUND_ANY_CHILD for placeholder [FINDING1] with CUI '21522001' and name 'abdominal colic finding'
		With phrase: 'Description: Acute appendicitis\nCC: [FIN [271 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING1] with CUI '21522001' and name 'abdomen pain'
		With phrase: 'Description: Acute appendicitis\nCC: [FIN [271 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING1] with CUI '21522001' and name 'colicky pain'
		With phrase: 'Description: Acute appendicitis\nCC: coli [271 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING1] with CUI '21522001' and name 'colicky pain'
		With phrase: 'Description: Acute appendicitis\nCC: coli [271 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING1] with CUI '21522001' and name 'colicky pain'
		With phrase: 'Description: Acute appendicitis\nCC: Abdo [291 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING3] with CUI '386661006' and name 'hyperthermia'
		With phrase: 'Description: Acute appendicitis\nCC: Abdo [295 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING3] with CUI '386661006' and name 'high temperature'
		With phrase: 'Description: Acute appendicitis\nCC: Abdo [295 chars] d Nausea. He denied Diarrhea.\n'
			FAIL for placeholder [FINDING3] with CUI '386661006' and name 'high temperature'
		With phrase: 'Description: Migraine with aura\nCC: Unil [340 chars] obia. He denied [NEGFINDING].\n'
			FAIL for placeholder [NEGFINDING] with CUI '386661006' and name 'hyperthermia'
			FAIL for placeholder [NEGFINDING] with CUI '386661006' and name 'high temperature'
		With phrase: 'Description: Acute appendicitis\nCC: Abdo [283 chars] usea. He denied [NEGFINDING].\n'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'loose stools'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'watery stool'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'loose bowel movement'
			FOUND_ANY_CHILD for placeholder [NEGFINDING] with CUI '62315008' and name 'diarrhea symptom'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'loose bowel motion'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'loose bowel motions'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'loose stool'
			FOUND_ANY_CHILD for placeholder [NEGFINDING] with CUI '62315008' and name 'diarrhea symptoms'
			FOUND_ANY_CHILD for placeholder [NEGFINDING] with CUI '62315008' and name 'diarrhea symptom finding'
			FAIL for placeholder [NEGFINDING] with CUI '62315008' and name 'watery stools'
		With phrase: 'Description: Epidemic vertigo\nCC: Severe [311 chars] usea. He denied [NEGFINDING].\n'
			FAIL for placeholder [NEGFINDING] with CUI '15188001' and name 'decreased hearing'
			FAIL for placeholder [NEGFINDING] with CUI '15188001' and name 'decreased hearing finding'
			FAIL for placeholder [NEGFINDING] with CUI '60862001' and name 'ringing in ear'
```

</details>

## The regression suite format

The format has some documentation in the default (`config/default_regression_tests.yml`).
One should refer to those for now.


