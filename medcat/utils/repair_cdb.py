import pandas as pd
from medcat.config import Config
from medcat.cat import CAT
from medcat.cdb_maker import CDBMaker


class RepairCDB(object):
    def __init__(self, base_cdb, final_cdb, vocab):
        self.base_cdb = base_cdb
        self.vocab = vocab
        self.final_cdb = final_cdb
        self.final_cat = None
        self.cdb = None
        self.cat = None
        self.base_cat = None

    def prepare(self, cuis):
        self.base_cdb.filter_by_cui(cuis)

        csv = [['cui', 'name']]
        names = set()
        cui = 0
        for base_cui in self.base_cdb.cui2names:
            if self.base_cdb.cui2context_vectors.get(base_cui, {}):
                for name in self.base_cdb.cui2names[base_cui]:
                    if name not in names and name in self.base_cdb.name2cuis:
                        csv.append([cui, name.replace("~", " ")])
                        cui += 1
                        names.add(name)

        df_cdb = pd.DataFrame(csv[1:], columns=csv[0])
        df_cdb.to_csv("/tmp/data.csv", index=False)

        config = Config()
        cdb_maker = CDBMaker(config=config)

        cdb = cdb_maker.prepare_csvs(['/tmp/data.csv'])

        # Rempove ambigous
        for name in cdb.name2cuis:
            cuis = cdb.name2cuis[name]
            if len(cuis) > 1:
                cnts = []
                for cui in cuis:
                    cnts.append([cui, len(cdb.cui2names[cui])])
                cnts.sort(key=lambda x: x[1])
                cdb.name2cuis[name] = [cnts[-1][0]]

        self.cdb = cdb
        self.base_cdb.reset_cui_count(n=10)
        self.cat = CAT(cdb=self.cdb, config=self.cdb.config, vocab=self.vocab)
        self.base_cat = CAT(cdb=self.base_cdb, config=self.base_cdb.config, vocab=self.vocab)

    def train(self, data_iterator, n_docs=100000):
        docs = []
        for doc in data_iterator:
            docs.append(doc)
            if len(docs) >= n_docs:
                break
        self.cat.train(data_iterator=docs)
        self.base_cat.train(data_iterator=docs)

    def calculate_scores(self, count_limit=1000):
        data = [['new_cui', 'base_cui', 'name', 'new_count', 'base_count', 'score', 'decision']]
        for name, cuis2 in self.cdb.name2cuis.items():
            cui2 = cuis2[0]
            count2 = self.cdb.cui2count_train.get(cui2, 0)
            if count2 > count_limit:
                cuis = self.base_cdb.name2cuis.get(name, [])
                for cui in cuis:
                    count = self.base_cdb.cui2count_train.get(cui, 0)
                    if self.base_cdb.cui2context_vectors.get(cui, {}):
                        score = count2 / count
                        data.append([cui2, cui, name, count2, count, score, ''])

        self.scores_df = pd.DataFrame(data[1:], columns=data[0])


    def unlink_names(self, sort='score', skip=0, cui_filter=None, apply_existing_decisions=0):
        scores_df = self.scores_df.sort_values(sort, ascending=False)
        self.final_cdb.config.general['full_unlink'] = False
        if self.final_cat is None:
            self.final_cat = CAT(cdb=self.final_cdb, config=self.final_cdb.config, vocab=self.vocab)

        for ind, row in enumerate(scores_df.iterrows()):
            row_ind, row = row
            if ind < skip:
                continue
            name = row['name']
            base_cui = row['base_cui']
            new_cui = row['new_cui']
            base_count = row['base_count']
            new_count = row['new_count']
            cui = row['base_cui']
            if base_cui in cui_filter:
                print("{:3} -- {:20} -> {:20}, base_count: {}, new_count: {}, cui: {}".format(
                    ind, str(name)[:20], str(self.final_cdb.get_name(base_cui))[:30], base_count, new_count, cui))

                if apply_existing_decisions and apply_existing_decisions > ind:
                    decision = row['decision']
                else:
                    decision = input("Decision (l/...): ")

                if decision == 'l':
                    names = self.cdb.cui2names[new_cui]
                    print("Unlinking: " + str(names))
                    print("\n\n")
                    for name in names:
                        self.final_cat.unlink_concept_name(base_cui, name, preprocessed_name=True)
                elif decision == 'f':
                    if base_cui in cui_filter:
                        print("Removing from filter: " + str(base_cui))
                        print("\n\n")
                        cui_filter.remove(base_cui)
                else:
                    decision = 'k' # Means keep

                self.scores_df.iat[row_ind, 6] = decision
