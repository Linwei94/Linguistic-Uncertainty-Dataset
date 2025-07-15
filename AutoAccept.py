import pandas as pd
from types import SimpleNamespace


class autoAccept:
    def __init__(self):
        pass

    def configProcess(self, row):
        self.constraints = {
            'constraint1': self.constraint1,
        }
        self.constraint1_config = self.getConstraint1Config(row)

    def getConstraint1Config(self, row):
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
        return SimpleNamespace(judgement=result_dict, threshold=4, )

    def constraint1(self, dataset, constraint1_config, constraintDesc='Over 4 valid in 5 sentences.'):
        results = {}
        for idx in constraint1_config.judgement.keys():
            row = dataset[dataset['index'] == idx]
            if row.empty:
                results[idx] = False
                continue
            ans = row.iloc[0]['answer']
            low, high = constraint1_config.judgement[idx]
            results[idx] = low <= ans <= high
        if sum(results.values()) >= constraint1_config.threshold:
            print("Constraint:" + constraintDesc + " PASS")
            return True
        print("Constraint:" + constraintDesc + " Falied")
        return False

    def runAllConstraints(self, params_dict, threshold=1.00):
        results = []
        for name, func in self.constraints.items():
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
        return ratio >= threshold

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
        data.loc[data['AssignmentId'] == assignment_id, 'Reject'] = 'Your response did not meet our quality standards.'
        return data

    def mainprocessFromCSV(self, resultFileName):
        df = pd.read_csv(resultFileName)
        df_fin = df
        print("Successfully read result csv.")
        for idx, row in df.iterrows():
            print()
            print(f"Dealing with {idx + 1}/{df.shape[0]} hits.")
            assignment_id = row['AssignmentId']
            data = self.resultToDataset(row)
            # 配置约束
            self.configProcess(row)
            params = {
                'constraint1': (data, self.constraint1_config,),
            }
            if self.runAllConstraints(params):
                try:
                    self.approveProcessFromCSV(assignment_id, df_fin)
                except Exception as e:
                    print(f"Error approving assignment {assignment_id}: {e}")
            else:
                try:
                    self.rejectProcessFromCSV(assignment_id, df_fin)
                except Exception as e:
                    print(f"Error rejecting assignment {assignment_id}: {e}")
        df_fin.to_csv(f"{resultFileName[:-4]}_Upload.csv", index=False)


def testMainProcessFromCSV():
    # 实例化
    aac = autoAccept()

    filename = 'Batch_416751_batch_results.csv'

    # 处理数据
    aac.mainprocessFromCSV(filename)


if __name__ == '__main__':
    testMainProcessFromCSV()
