import pandas as pd
from types import SimpleNamespace
import argparse


class autoAccept:
    def __init__(self):
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

    def constrain1(self, dataset, constrain1_config, fuzzyBoundary=0.00, constrainDesc='Over 4 valid in 5 sentences.'):
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
            return True
        print("constrain:" + constrainDesc + " Falied")
        return False

    def getconstrain2Config(self, ):
        return

    def constrain2(self, row, allSentenceFilePath, consistencyBar, constrainDesc='Selected Consistency Rate'):
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
            return True
        print("constrain:" + constrainDesc + " Falied")
        return False

    def constrain3(self, constrainDesc='Total z-score'):
        return True

    def constrain4(self, constrainDesc='Individual z-score'):
        return True

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
        return ratio >= all_constrains_pass_rate and all(x > y for x, y in zip(results, all_constrains_pass_boolean_list))

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
    ) -> None:
        df = pd.read_csv(resultFileName)
        df_fin = df
        print("Successfully read result csv.")
        resrow = {
            'pass_row': [],
            'fail_row': [],
        }
        for idx, row in df.iterrows():
            print()
            print(f"Dealing with {idx + 1}/{df.shape[0]} hits.")
            assignment_id = row['AssignmentId']
            # constrain1 data
            data = self.resultToDataset(row)
            # constrain2 data
            # no other data required except row
            # constrain3 data
            
            # pass
            # constrains configue
            self.configProcess(row, constrain1_pass_count)
            params = {
                'constrain1': (data, self.constrain1_config,),
                'constrain2': (row, constrain2_all_sentence_filename, constrain2_consistency_bar, ),
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
        print(f"total pass {len(resrow['pass_row'])}\n total fail {len(resrow['fail_row'])}")
        df_fin.to_csv(f"{resultFileName[:-4]}_Upload.csv", index=False)


def testMainProcessFromCSV(args):
    aac = autoAccept()

    aac.main_process_from_CSV(args.filename,
                              args.all_constrains_pass_rate,
                              args.all_constrains_pass_boolean_list,
                              args.constrain1_pass_count,
                              args.constrain2_all_sentence_filename,
                              args.constrain2_consistency_bar)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Auto Accept Script')
    # public
    argparser.add_argument('--filename',
                           type=str,
                           default='res/Batch_5332788_batch_results.csv',
                           help='Path to the CSV file containing results')
    argparser.add_argument('--all_constrains_pass_rate',
                           type=float,
                           default=0.5,
                           help='The ratio required to pass all constraints')
    argparser.add_argument('--all_constrains_pass_boolean_list',
                           type=list,
                           default=[False, False, False, False],
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
    #
    args = argparser.parse_args()
    #
    testMainProcessFromCSV(args)
