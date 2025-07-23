import pandas as pd
from types import SimpleNamespace
import argparse


class autoAccept:
    def __init__(self):
        self.constrain1_boolean = []
        self.constrain2_boolean = []
        self.constrain3_boolean = []
        self.constrain4_boolean = []
        pass

    def configProcess(self, row, pass_count):
        self.constrains = {
            'constrain1': self.constrain1,
            'constrain2': self.constrain2,
            'constrain3': self.constrain3,
            'constrain4': self.constrain4,
        }
        self.constrain1_config = self.getconstrain1Config(row, pass_count)
        self.constrain2_config = self.getconstrain2Config()

    def getconstrain1Config(self, row, pass_count):
        cols = [
            'Input.val_upper_bound_1', 'Input.val_lower_bound_1',
            'Input.val_upper_bound_2', 'Input.val_lower_bound_2',
            'Input.val_upper_bound_3', 'Input.val_lower_bound_3',
            'Input.val_upper_bound_4', 'Input.val_lower_bound_4',
            'Input.val_upper_bound_5', 'Input.val_lower_bound_5',
        ]
        result_dict = {}
        numbers = sorted(set(col.split('_')[-1] for col in cols))

        for num in numbers:
            key = f"Answer.confidence_score_val_sentence_{num}"
            lower_col = f"Input.val_lower_bound_{num}"
            upper_col = f"Input.val_upper_bound_{num}"

            result_dict[key] = [row[lower_col], row[upper_col]]
        return SimpleNamespace(judgement=result_dict, threshold=pass_count, )

    # Check the result by whether it is consistent with the valued sentence
    def constrain1(self, dataset, constrain1_config, fuzzyBoundary=0.00, constrainDesc='valid sentence check.'):
        results = {}
        for idx in constrain1_config.judgement.keys():
            row = dataset[dataset['index'] == idx]
            if row.empty:
                results[idx] = False
                continue
            ans = row.iloc[0]['answer']
            low, high = constrain1_config.judgement[idx]
            results[idx] = low * \
                (1 - fuzzyBoundary) <= ans <= high * (1 + fuzzyBoundary)
        if sum(results.values()) >= constrain1_config.threshold:
            print("constrain:" + constrainDesc + " PASS")
            self.constrain1_boolean.append(True)
            return True
        print("constrain:" + constrainDesc + " Falied")
        self.constrain1_boolean.append(False)
        return False

    def getconstrain2Config(self, ):
        return

    # Check the consistency of all questions with the AI to see if they meet the conditions
    def constrain2(
        self,
        row: pd.Series,
        allSentenceFilePath: str,
        consistencyBar: float,
        constrainDesc: str = 'Selected Consistency Rate'
    ) -> bool:
        confidence_bounds = {
            'completely uncertain': (0, 30),
            'lowest': (10, 50),
            'low': (20, 70),
            'moderate': (40, 90),
            'high': (60, 100)
        }
        all_sent_df = pd.read_csv(allSentenceFilePath)
        curr_success = 0
        for i in range(1, 101):
            llm_confidence_level = all_sent_df['confidence'][row[f'Input.index_{i}']]
            lower_bound, upper_bound = confidence_bounds[llm_confidence_level]
            if lower_bound <= row[f'Answer.confidence_score_sentence_{i}'] <= upper_bound:
                curr_success += 1
        if curr_success / 100.0 >= consistencyBar:
            print("constrain:" + constrainDesc + " PASS")
            self.constrain2_boolean.append(True)
            return True
        print("constrain:" + constrainDesc + " Falied")
        self.constrain2_boolean.append(False)
        return False

    # Check whether the conditions are met through the total offset deviation from the center in each dimension
    def constrain3(
        self,
        row: pd.Series,
        zscore_sum_threshold: int,
        constrainDesc='Total z-score'
    ) -> bool:
        zscores = []
        for i in range(1, 6):
            val_idx = row[f"Input.val_index_{i}"]
            score = row[f"Answer.confidence_score_val_sentence_{i}"]
            stats = self.stats_map.get(val_idx)
            if stats is None or stats['std'] == 0:
                zscore = 0
            else:
                zscore = (score - stats['mean']) / stats['std']
            zscores.append(zscore)
        zscore_sum = sum(abs(z) for z in zscores)
        if zscore_sum <= zscore_sum_threshold:
            print("constrain:" + constrainDesc + " PASS")
            self.constrain3_boolean.append(True)
            return True
        print("constrain:" + constrainDesc + " Falied")
        self.constrain3_boolean.append(False)
        return False

    # From the aspect of the valued sentence, check the validity by whether it deviates from the center
    def constrain4(
        self,
        row: pd.Series,
        zscore_value_threshold: float,
        zscore_count_threshold: int,
        constrainDesc: str = 'Individual z-score',
    ) -> bool:
        zscores = []
        for i in range(1, 6):
            val_idx = row[f"Input.val_index_{i}"]
            score = row[f"Answer.confidence_score_val_sentence_{i}"]
            stats = self.stats_map.get(val_idx)
            if stats is None or stats['std'] == 0:
                zscore = 0
            else:
                zscore = (score - stats['mean']) / stats['std']
            zscores.append(zscore)
        high_zscore_count = sum(
            abs(z) > zscore_value_threshold for z in zscores)
        if high_zscore_count <= zscore_count_threshold:
            print("constrain:" + constrainDesc + " PASS")
            self.constrain4_boolean.append(True)
            return True
        print("constrain:" + constrainDesc + " Falied")
        self.constrain4_boolean.append(False)
        return False

    def runAllconstrains(self, params_dict, all_constrains_pass_rate, all_constrains_pass_boolean_list):
        results = []
        for name, func in self.constrains.items():
            params = params_dict.get(name, ())
            if isinstance(params, dict):
                res = func(**params)
            elif isinstance(params, tuple) or isinstance(params, list):
                res = func(*params)
            else:
                res = func(params)
            results.append(res)
        true_count = sum(results)
        ratio = true_count / len(results) if results else 0
        return ratio >= all_constrains_pass_rate and all(x >= y for x, y in zip(results, all_constrains_pass_boolean_list))

    def resultToDataset(self, df):
        cols = ['Answer.confidence_score_val_sentence_1', 'Answer.confidence_score_val_sentence_2',
                'Answer.confidence_score_val_sentence_3', 'Answer.confidence_score_val_sentence_4',
                'Answer.confidence_score_val_sentence_5']
        data = []
        for col in cols:
            val = df[col]
            data.append((col, val))

        data.sort(key=lambda x: x[0])
        result_df = pd.DataFrame(data, columns=['index', 'answer'])
        return result_df

    def approveProcessFromCSV(self, assignment_id, data):
        data.loc[data['AssignmentId'] == assignment_id, 'Approve'] = 'x'
        print(f"Approved assignment: {assignment_id} ")
        return data

    def rejectProcessFromCSV(self, assignment_id, data):
        data.loc[data['AssignmentId'] == assignment_id,
                 'Reject'] = 'Your response did not meet our quality standards.'
        return data

    def mean_std_for_each_val_index(
        self,
        filepath: str,
    ) -> dict:
        df = pd.read_csv(filepath)
        index_cols = [f"Input.val_index_{i}" for i in range(1, 6)]
        score_cols = [
            f"Answer.confidence_score_val_sentence_{i}" for i in range(1, 6)]
        missing = set(index_cols + score_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        val_index_df = df[index_cols].melt(value_name='val_index')
        score_df = df[score_cols].melt(value_name='score')
        combined = pd.concat(
            [val_index_df['val_index'], score_df['score']], axis=1)
        stats_map = (
            combined
            .groupby('val_index')['score']
            .agg(['mean', 'std'])
            .to_dict('index')
        )
        return stats_map

    # predict main process logic:
    # 1. load global data
    # 2. prepare data from all data
    # 3. for each row, apply every constraints
    # 4. generate upload files.

    def main_process_from_CSV(
        self,
        resultFileName: str,
        all_constrains_pass_rate: float,
        all_constrains_pass_boolean_list: list,
        constrain1_pass_count: int,
        constrain2_all_sentence_filename: str,
        constrain2_consistency_bar: float,
        constrain3_pass_count_for_sum_zscore: int,
        constrain4_pass_count_for_single_value: float,
        constrain4_pass_count: int,
    ) -> None:
        df = pd.read_csv(resultFileName)
        df_fin = df
        print("Successfully read result csv.")
        resrow = {
            'pass_row': [],
            'fail_row': [],
        }
        # prepare golbal data for constrain
        self.stats_map = self.mean_std_for_each_val_index(resultFileName)
        # start check every assignment
        for idx, row in df.iterrows():
            print()
            print(f"Dealing with {idx + 1}/{df.shape[0]} hits.")
            assignment_id = row['AssignmentId']
            # constrain1 data
            data = self.resultToDataset(row)
            # constrain2 data
            # no other data required except row
            # constrain3 data
            # no other data required except row which needed to prepare here.
            # constrain4 data
            # no other data required except row which needed to prepare here.
            # constraints configue
            self.configProcess(row, constrain1_pass_count)
            params = {
                'constrain1': (data, self.constrain1_config,),
                'constrain2': (row, constrain2_all_sentence_filename, constrain2_consistency_bar,),
                'constrain3': (row, constrain3_pass_count_for_sum_zscore,),
                'constrain4': (row, constrain4_pass_count_for_single_value, constrain4_pass_count,),
            }
            # run constrains
            if self.runAllconstrains(params, all_constrains_pass_rate, all_constrains_pass_boolean_list):
                resrow['pass_row'].append(idx)
                try:
                    self.approveProcessFromCSV(assignment_id, df_fin)
                except Exception as e:
                    print(f"Error approving assignment {assignment_id}: {e}")
            else:
                resrow['fail_row'].append(idx)
                try:
                    self.rejectProcessFromCSV(assignment_id, df_fin)
                except Exception as e:
                    print(f"Error rejecting assignment {assignment_id}: {e}")
        print(
            f"total pass {len(resrow['pass_row'])}\n total fail {len(resrow['fail_row'])}")
        df_fin.to_csv(f"{resultFileName[:-4]}_Upload.csv", index=False)


def testMainProcessFromCSV(args):
    aac = autoAccept()

    aac.main_process_from_CSV(args.filename,
                              args.all_constrains_pass_rate,
                              args.all_constrains_pass_boolean_list,
                              args.constrain1_pass_count,
                              args.constrain2_all_sentence_filename,
                              args.constrain2_consistency_bar,
                              args.constrain3_pass_count_for_sum_zscore,
                              args.constrain4_pass_count_for_single_value,
                              args.constrain4_pass_count,
                              )
    # print("constrain1:", aac.constrain1_boolean)
    # print("constrain2:", aac.constrain2_boolean)
    # print("constrain3:", aac.constrain3_boolean)
    # print("constrain4:", aac.constrain4_boolean)

    # save results
    import pickle
    with open('constrain1_boolean.pkl', 'wb') as f:
        pickle.dump(aac.constrain1_boolean, f)
    with open('constrain2_boolean.pkl', 'wb') as f:
        pickle.dump(aac.constrain2_boolean, f)
    with open('constrain3_boolean.pkl', 'wb') as f:
        pickle.dump(aac.constrain3_boolean, f)
    with open('constrain4_boolean.pkl', 'wb') as f:
        pickle.dump(aac.constrain4_boolean, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Auto Accept Script')
    # public
    argparser.add_argument('--filename',
                           type=str,
                           default='tmp/Batch_5332788_batch_results.csv',
                           help='Path to the CSV file containing results')
    argparser.add_argument('--all_constrains_pass_rate',
                           type=float,
                           default=0.75,
                           help='The ratio required to pass all constraints')
    argparser.add_argument('--all_constrains_pass_boolean_list',
                           type=list,
                           default=[False, True, True, True],
                           help='The boolean list required to pass all constraints')
    # constrain 1
    argparser.add_argument('--constrain1_pass_count',
                           type=int,
                           default=4,
                           help='Number of sentences that must pass the constrain')
    # constrain 2
    argparser.add_argument('--constrain2_all_sentence_filename',
                           type=str,
                           default='all_sentences_by_confidence.csv',
                           help='Path to all_sentences_by_confidence.csv')
    argparser.add_argument('--constrain2_consistency_bar',
                           type=float,
                           default=0.45,
                           help='Lower bound of consistency distribution obtained by human through graph')
    # constrain 3
    argparser.add_argument('--constrain3_pass_count_for_sum_zscore',
                           type=int,
                           default=8,
                           help='hyperparameter that the total error of zscores of the five valued sentences ')
    # constrain 4
    argparser.add_argument('--constrain4_pass_count_for_single_value',
                           type=float,
                           default=0.8,
                           help='hyperparameter that error for a valued sentence to pass')
    # constrain 4
    argparser.add_argument('--constrain4_pass_count',
                           type=int,
                           default=4,
                           help='hyperparameter that the number of sentence to pass the constrain')
    #
    args = argparser.parse_args()
    #
    testMainProcessFromCSV(args)
