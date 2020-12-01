from medcat.utils.medmentions import original2concept_csv
from medcat.utils.medmentions import original2json
from medcat.utils.medmentions import original2pure_text

_ = original2json("../../examples/medmentions/medmentions.txt", '../../examples/medmentions/tmp_medmentions.json')
_ = original2concept_csv("../../examples/medmentions/medmentions.txt", '../../examples/medmentions/tmp_medmentions.csv')
original2pure_text("../../examples/medmentions/medmentions.txt", '../../examples/medmentions/tmp_medmentions_text_only.txt')
