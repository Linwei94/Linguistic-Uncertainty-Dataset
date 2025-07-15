import boto3
import pandas as pd
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

"""
autoAccept类包含连接aws，注册Qualification，配置约束和处理result
在开发环境中按照main()运行即可，
在生产环境中，需要先完成连接aws，注册Qualification,以及配置约束，将注册Qualification的数据(该数据注册后自动保存)，
等问卷收集结束并下载csv
然后运行处理result;
"""


class autoAccept:
    def __init__(self):
        self.mturk = None

    def connectProcess(self, login_info):
        self.connectAws(login_info)

    def registerProcess(self, qualificationTypeParams):
        self.qualificationTypeId = self.createQualificationType(qualificationTypeParams)

    def configProcess(self):
        with open("qualification.json", "r") as f:
            self.qualificationTypeId = json.load(f)

        self.constraints = {
            'constraint1': self.constraint1,
            # 'constraint2': self.constraint2,
        }
        self.constraint1_config = SimpleNamespace(
            judgement={
                '20001': [75, 100],
                '20002': [25, 75],
                '20003': [65, 100],
                '20004': [65, 100],
                '20005': [0, 65]
            },
            threshold=4,
        )
        # self.constrain1_config = self.getConstrain1ConfigFromCSV('')
        # self.constraint2_config = SimpleNamespace(
        #     workerAssignmentCnt={},
        #     QualificationTypeId=self.qualificationTypeId['Max5Assignments'],
        #     threshold=5,
        # )

    def connectAws(self, login_info):
        self.mturk = boto3.client(
            'mturk',
            aws_access_key_id=login_info['aws_access_key_id'],
            aws_secret_access_key=login_info['aws_secret_access_key'],
            region_name=login_info['region_name'],
            endpoint_url=login_info['endpoint_url']
        )
        print("Account balance:", self.mturk.get_account_balance()['AvailableBalance'])
        return self.mturk

    def createQualificationType(self, qualificationTypeParams):
        qualification_type_id = {}
        for param in qualificationTypeParams:
            try:
                response = self.mturk.create_qualification_type(
                    Name=param['Name'],
                    Keywords=param['Keywords'],
                    Description=param['Description'],
                    QualificationTypeStatus=param['QualificationTypeStatus'],
                )
                qualification_type_id[param['Name']] = response['QualificationType']['QualificationTypeId']
                print(f"Successly create qualification type {param['Name']}: {qualification_type_id[param['Name']]}")
            except Exception as e:
                print(f"Error create qualification type {param['Name']}: {e}")
                qualification_type_id[param['Name']] = None

        with open("qualification.json", "w") as f:
            json.dump(qualification_type_id, f, indent=2)
        return qualification_type_id

    def constraint1(self, dataset, constraint1_config, constraintDesc='Over 4 valid in 5 sentences.'):
        results = {}
        for idx in constraint1_config.judgement.keys():
            row = dataset[dataset['index'] == int(idx)]
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

    # def constraint2(self, worker_id, constraint2_config, constraintDesc='Every worker\'s <= 5.'):
    #     count = constraint2_config.workerAssignmentCnt.get(worker_id, 0)
    #     if count >= constraint2_config.threshold:
    #         print("Constraint:" + constraintDesc + " Falied")
    #         return False
    #     constraint2_config.workerAssignmentCnt[worker_id] = count + 1
    #     try:
    #         self.mturk.associate_qualification_with_worker(
    #             QualificationTypeId=constraint2_config.QualificationTypeId,
    #             WorkerId=worker_id,
    #             IntegerValue=constraint2_config.workerAssignmentCnt[worker_id],
    #             SendNotification=False
    #         )
    #     except Exception as e:
    #         print(f"[Qualification] Failed assigning: {e}")
    #     print("Constraint:" + constraintDesc + " PASS")
    #     return True

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
        input_index_cols = [f'Answer.index_{i}' for i in range(1, 121)]
        input_val_index_cols = [f'Input.val_index_{i}' for i in range(1, 6)]

        index_values_1 = df[input_index_cols].values.flatten()
        index_values_2 = df[input_val_index_cols].values.flatten()

        index_all = pd.Series(list(index_values_1) + list(index_values_2))

        answer_conf_cols = [f'Answer.confidence_score_sentence_{i}' for i in range(1, 121)]
        answer_val_cols = [f'Answer.confidence_score_val_sentence_{i}' for i in range(1, 6)]

        answer_values_1 = df[answer_conf_cols].values.flatten()
        answer_values_2 = df[answer_val_cols].values.flatten()

        answer_all = pd.Series(list(answer_values_1) + list(answer_values_2))

        result_df = pd.DataFrame({
            'index': index_all,
            'answer': answer_all
        })
        return result_df

    # Note:
    # development environment : False
    # production environment : True
    def approveProcess(self, assignment_id, flag=False):
        if flag == True:
            response = self.mturk.approve_assignment(
                AssignmentId=assignment_id,
                RequesterFeedback="Thank you for your work!",
                OverrideRejection=False
            )
            print(response)
            print(f"Approved assignment online: {assignment_id} ")
        elif flag == False:
            print(f"Approved assignment offline: {assignment_id}")
        return

    def approveProcessFromCSV(self, assignment_id, data):
        data.loc[data['AssignmentId'] == assignment_id, 'Approve'] = 'x'
        print(f"Approved assignment: {assignment_id} ")
        return data

    # Note:
    # development environment : False
    # production environment : True
    def rejectProcess(self, assignment_id, flag=False):
        if flag == True:
            response = self.mturk.reject_assignment(
                AssignmentId=assignment_id,
                RequesterFeedback="Your response did not meet our quality standards."
            )
            print(response)
            print(f"Rejected assignment online: {assignment_id}")
        elif flag == False:
            print(f"Rejected assignment offline: {assignment_id}")
        return

    def rejectProcessFromCSV(self, assignment_id, data):
        data.loc[data['AssignmentId'] == assignment_id, 'Reject'] = 'Your response did not meet our quality standards.'
        print(f"Rejected assignment : {assignment_id}")
        return data

    def mainprocess(self, resultFileName):
        df = pd.read_csv(resultFileName)
        print("Successfully read result csv.")
        for idx, row in df.iterrows():
            print()
            print(f"Dealing with {idx + 1}/{df.shape[0]} hits.")
            hit_id = row['HITId']
            assignments = self.mturk.list_assignments_for_hit(
                HITId=hit_id,
                AssignmentStatuses=['Submitted']
            )['Assignments']
            for assignment in assignments:
                assignment_id = assignment['AssignmentId']
                worker_id = assignment['WorkerId']
                data = self.resultToDataset(row)
                params = {
                    'constraint1': (data, self.constraint1_config,),
                    'constraint2': (worker_id, self.constraint2_config,),
                }
                if self.runAllConstraints(params):
                    try:
                        self.approveProcess(assignment_id, True)
                    except Exception as e:
                        print(f"Error approving assignment {assignment_id}: {e}")
                else:
                    try:
                        self.rejectProcess(assignment_id, True)
                    except Exception as e:
                        print(f"Error rejecting assignment {assignment_id}: {e}")

    def mainprocessFromCSV(self, resultFileName):
        df = pd.read_csv(resultFileName)
        df_fin = df
        print("Successfully read result csv.")
        for idx, row in df.iterrows():
            print()
            print(f"Dealing with {idx + 1}/{df.shape[0]} hits.")
            hit_id = row['HITId']
            assignment_id = row['AssignmentId']
            worker_id = row['WorkerId']
            data = self.resultToDataset(row)
            params = {
                'constraint1': (data, self.constraint1_config,),
                # 'constraint2': (worker_id, self.constraint2_config,),
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

    def getAllHits(self):
        next_token = None
        all_hits = []
        page = 1
        while True:
            print(f"Fetching HITs page {page}...")
            if next_token:
                response = self.mturk.list_hits(NextToken=next_token, MaxResults=100)
            else:
                response = self.mturk.list_hits(MaxResults=100)

            all_hits.extend(response['HITs'])

            next_token = response.get('NextToken')
            if not next_token:
                break
            page += 1

        print(f"Total HITs fetched: {len(all_hits)}")
        return all_hits

    def getAllAssignments(self, hit_id):
        all_assignments = []
        next_token = None
        while True:
            if next_token:
                response = self.mturk.list_assignments_for_hit(
                    HITId=hit_id,
                    MaxResults=100,
                    AssignmentStatuses=['Submitted', 'Approved', 'Rejected'],  # 你可以按状态过滤
                    NextToken=next_token
                )
            else:
                response = self.mturk.list_assignments_for_hit(
                    HITId=hit_id,
                    MaxResults=100
                )
            all_assignments.extend(response['Assignments'])
            next_token = response.get('NextToken')
            if not next_token:
                break
        return all_assignments

    def api2Dataset(self, answer_xml):
        """把Answer字段从XML格式解析成字典"""
        ns = {
            'ns': 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd'}
        root = ET.fromstring(answer_xml)
        result = {}
        for answer in root.findall('ns:Answer', ns):
            qid = answer.find('ns:QuestionIdentifier', ns).text
            val = answer.find('ns:FreeText', ns).text
            result[f'Answer.{qid}'] = val
        return self.xmlAnswerToDataset(result)

    def xmlAnswerToDataset(self, answer_dict):
        # 解析 index 值
        index_values_1 = [answer_dict.get(f'Answer.index_{i}', None) for i in range(1, 121)]
        index_values_2 = [answer_dict.get(f'Answer.val_index_{i}', None) for i in range(1, 6)]
        index_all = pd.Series(index_values_1 + index_values_2)

        # 解析 answer 值
        answer_values_1 = [answer_dict.get(f'Answer.confidence_score_sentence_{i}', None) for i in range(1, 121)]
        answer_values_2 = [answer_dict.get(f'Answer.confidence_score_val_sentence_{i}', None) for i in range(1, 6)]
        answer_all = pd.Series(answer_values_1 + answer_values_2)

        # 构建 DataFrame
        result_df = pd.DataFrame({
            'index': index_all,
            'answer': answer_all
        })

        # 可选：去掉空行
        result_df = result_df.dropna().reset_index(drop=True)

        return result_df

    def dealFromApi(self, title='Test1'):
        hits = self.getAllHits()
        for hit in hits:
            if hit['Title'] == title:
                assignments = self.getAllAssignments(hit['HITId'])
                if assignments != []:
                    for assignment in assignments:
                        assignment_id = assignment['AssignmentId']
                        worker_id = assignment['WorkerId']
                        data = self.api2Dataset(assignment['Answer'])
                        params = {
                            'constraint1': (data, self.constraint1_config,),
                            # 'constraint2': (worker_id, self.constraint2_config,),
                        }
                        if self.runAllConstraints(params):
                            try:
                                self.approveProcess(assignment_id, True)
                            except Exception as e:
                                print(f"Error approving assignment {assignment_id}: {e}")
                        else:
                            try:
                                self.rejectProcess(assignment_id, True)
                            except Exception as e:
                                print(f"Error rejecting assignment {assignment_id}: {e}")


def main():
    # 实例化
    aac = autoAccept()

    # 连接aws
    login_info = {
        'aws_access_key_id': '',
        'aws_secret_access_key': '',
        'region_name': 'us-east-1',
        'endpoint_url': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    }
    aac.connectProcess(login_info)

    # 注册qualification
    qualificationTypeParams = [
        {
            'Name': "Max5Assignments",
            'Keywords': "max5, control, limit",
            'Description': "Limits a worker to at most 5 assignments.",
            'QualificationTypeStatus': "Active",
        }
    ]
    aac.registerProcess(qualificationTypeParams)

    # 配置约束
    aac.configProcess()

    # 处理数据
    filename = 'Batch_416679_batch_results.csv'
    aac.mainprocess(filename)


def testMainProcessFromCSV():
    # 实例化
    aac = autoAccept()

    # # 连接aws
    # login_info = {
    #     'aws_access_key_id': '',
    #     'aws_secret_access_key': '',
    #     'region_name': 'us-east-1',
    #     'endpoint_url': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    # }
    # aac.connectProcess(login_info)

    # # 注册qualification
    # qualificationTypeParams = [
    #     {
    #         'Name': "Max5Assignments",
    #         'Keywords': "max5, control, limit",
    #         'Description': "Limits a worker to at most 5 assignments.",
    #         'QualificationTypeStatus': "Active",
    #     }
    # ]
    # aac.registerProcess(qualificationTypeParams)

    # 配置约束
    aac.configProcess()

    # 处理数据
    filename = 'Batch_416746_batch_results.csv'
    aac.mainprocessFromCSV(filename)


def testDealFromApi():
    aac = autoAccept()

    # 连接aws
    login_info = {
        'aws_access_key_id': '',
        'aws_secret_access_key': '',
        'region_name': 'us-east-1',
        'endpoint_url': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    }
    aac.connectProcess(login_info)
    # 配置约束
    aac.configProcess()
    aac.dealFromApi("Test2")


if __name__ == '__main__':
    # main()
    # testDealFromApi()
    testMainProcessFromCSV()